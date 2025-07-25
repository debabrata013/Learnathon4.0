#!/usr/bin/env python3
"""
ML Fraud Detection Preprocessing Pipeline - Part 2
=================================================
Feature Engineering, Selection, and Normalization
"""

# Continuation of FraudDetectionPreprocessor class

    def encode_categorical_features(self) -> None:
        """Advanced categorical feature encoding"""
        logger.info("Encoding categorical features...")
        
        categorical_cols = self.combined_train.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'Fraud_Ind']  # Exclude target
        
        encoding_info = {}
        
        for col in categorical_cols:
            unique_values = self.combined_train[col].nunique()
            
            if unique_values <= 10:  # Use Label Encoding for low cardinality
                le = LabelEncoder()
                self.combined_train[col] = le.fit_transform(self.combined_train[col].astype(str))
                self.label_encoders[col] = le
                encoding_info[col] = {'method': 'label_encoding', 'unique_values': unique_values}
                
            else:  # Use frequency encoding for high cardinality
                freq_map = self.combined_train[col].value_counts().to_dict()
                self.combined_train[col] = self.combined_train[col].map(freq_map)
                encoding_info[col] = {'method': 'frequency_encoding', 'unique_values': unique_values}
        
        # Encode target variable
        if 'Fraud_Ind' in self.combined_train.columns:
            le_target = LabelEncoder()
            self.combined_train['Fraud_Ind'] = le_target.fit_transform(self.combined_train['Fraud_Ind'])
            self.label_encoders['Fraud_Ind'] = le_target
            encoding_info['Fraud_Ind'] = {'method': 'label_encoding', 'classes': list(le_target.classes_)}
        
        self.preprocessing_summary['categorical_encoding'] = encoding_info
        logger.info(f"Categorical encoding completed for {len(categorical_cols)} columns")
    
    def select_important_features(self, n_features: int = 15) -> List[str]:
        """Advanced feature selection using multiple methods"""
        logger.info(f"Selecting top {n_features} important features...")
        
        # Prepare features and target
        if 'Fraud_Ind' not in self.combined_train.columns:
            logger.error("Target variable 'Fraud_Ind' not found")
            return []
        
        X = self.combined_train.drop(['Fraud_Ind'], axis=1)
        y = self.combined_train['Fraud_Ind']
        
        # Remove non-numeric columns that couldn't be encoded
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        feature_scores = {}
        
        # Method 1: F-statistic
        try:
            f_selector = SelectKBest(score_func=f_classif, k='all')
            f_selector.fit(X_numeric, y)
            f_scores = dict(zip(X_numeric.columns, f_selector.scores_))
            feature_scores['f_statistic'] = f_scores
        except Exception as e:
            logger.warning(f"F-statistic feature selection failed: {e}")
        
        # Method 2: Mutual Information
        try:
            mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
            mi_selector.fit(X_numeric, y)
            mi_scores = dict(zip(X_numeric.columns, mi_selector.scores_))
            feature_scores['mutual_info'] = mi_scores
        except Exception as e:
            logger.warning(f"Mutual information feature selection failed: {e}")
        
        # Method 3: Correlation with target
        try:
            correlations = X_numeric.corrwith(y).abs()
            corr_scores = correlations.to_dict()
            feature_scores['correlation'] = corr_scores
        except Exception as e:
            logger.warning(f"Correlation feature selection failed: {e}")
        
        # Combine scores (average ranking)
        all_features = set()
        for method_scores in feature_scores.values():
            all_features.update(method_scores.keys())
        
        combined_scores = {}
        for feature in all_features:
            scores = []
            for method, method_scores in feature_scores.items():
                if feature in method_scores:
                    # Normalize scores to 0-1 range
                    max_score = max(method_scores.values())
                    min_score = min(method_scores.values())
                    if max_score != min_score:
                        normalized_score = (method_scores[feature] - min_score) / (max_score - min_score)
                    else:
                        normalized_score = 1.0
                    scores.append(normalized_score)
            
            combined_scores[feature] = np.mean(scores) if scores else 0
        
        # Select top features
        selected_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:n_features]
        self.selected_features = [feature[0] for feature in selected_features]
        
        # Store feature analysis
        self.feature_analysis = {
            'feature_scores': feature_scores,
            'combined_scores': combined_scores,
            'selected_features': self.selected_features,
            'selection_method': 'Combined ranking (F-statistic, Mutual Info, Correlation)'
        }
        
        logger.info(f"Feature selection completed. Selected features: {self.selected_features}")
        return self.selected_features
    
    def normalize_features(self) -> None:
        """Normalize selected features using StandardScaler"""
        logger.info("Normalizing selected features...")
        
        if not self.selected_features:
            logger.error("No features selected for normalization")
            return
        
        # Prepare data with selected features
        X_selected = self.combined_train[self.selected_features]
        
        # Fit and transform the scaler
        X_normalized = self.scaler.fit_transform(X_selected)
        
        # Update the dataframe
        for i, feature in enumerate(self.selected_features):
            self.combined_train[f"{feature}_normalized"] = X_normalized[:, i]
        
        # Store normalization info
        self.preprocessing_summary['normalization'] = {
            'method': 'StandardScaler',
            'features_normalized': self.selected_features,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist()
        }
        
        logger.info(f"Feature normalization completed for {len(self.selected_features)} features")
    
    def create_engineered_features(self) -> List[str]:
        """Create 5 new engineered features from selected features"""
        logger.info("Creating 5 new engineered features...")
        
        if not self.selected_features:
            logger.error("No selected features available for engineering")
            return []
        
        engineered_features = []
        
        try:
            # Feature 1: Claim to Premium Ratio
            if 'Total_Claim' in self.selected_features and 'Policy_Premium' in self.selected_features:
                self.combined_train['Claim_Premium_Ratio'] = (
                    self.combined_train['Total_Claim'] / 
                    (self.combined_train['Policy_Premium'] + 1)  # Add 1 to avoid division by zero
                )
                engineered_features.append('Claim_Premium_Ratio')
            
            # Feature 2: Age Risk Score
            if 'Age_Insured' in self.selected_features:
                # Higher risk for very young (<25) and older (>65) drivers
                self.combined_train['Age_Risk_Score'] = self.combined_train['Age_Insured'].apply(
                    lambda x: 2 if x < 25 or x > 65 else 1 if x < 30 or x > 60 else 0
                )
                engineered_features.append('Age_Risk_Score')
            
            # Feature 3: Vehicle Value to Claim Ratio
            if 'Vehicle_Cost' in self.selected_features and 'Total_Claim' in self.selected_features:
                self.combined_train['Vehicle_Claim_Ratio'] = (
                    self.combined_train['Total_Claim'] / 
                    (self.combined_train['Vehicle_Cost'] + 1)
                )
                engineered_features.append('Vehicle_Claim_Ratio')
            
            # Feature 4: Claim Complexity Score
            claim_components = ['Injury_Claim', 'Property_Claim', 'Vehicle_Claim']
            available_components = [col for col in claim_components if col in self.selected_features]
            
            if len(available_components) >= 2:
                # Count non-zero claim components
                self.combined_train['Claim_Complexity_Score'] = 0
                for component in available_components:
                    self.combined_train['Claim_Complexity_Score'] += (
                        self.combined_train[component] > 0).astype(int)
                engineered_features.append('Claim_Complexity_Score')
            
            # Feature 5: Time-based Risk Score
            date_columns = ['Policy_Start_Date', 'Accident_Date', 'Claims_Date']
            available_dates = [col for col in date_columns if col in self.combined_train.columns]
            
            if len(available_dates) >= 2:
                # Convert to datetime if not already
                for col in available_dates:
                    if self.combined_train[col].dtype == 'object':
                        self.combined_train[col] = pd.to_datetime(self.combined_train[col], errors='coerce')
                
                # Calculate days between policy start and accident
                if 'Policy_Start_Date' in available_dates and 'Accident_Date' in available_dates:
                    self.combined_train['Days_Policy_To_Accident'] = (
                        self.combined_train['Accident_Date'] - self.combined_train['Policy_Start_Date']
                    ).dt.days
                    
                    # Risk score: higher risk for very quick claims (< 30 days)
                    self.combined_train['Time_Risk_Score'] = self.combined_train['Days_Policy_To_Accident'].apply(
                        lambda x: 2 if x < 30 else 1 if x < 90 else 0
                    )
                    engineered_features.append('Time_Risk_Score')
        
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
        
        # If we couldn't create 5 features, create simple mathematical combinations
        while len(engineered_features) < 5 and len(self.selected_features) >= 2:
            try:
                # Create interaction features from top selected features
                feature_idx = len(engineered_features)
                if feature_idx < len(self.selected_features) - 1:
                    feat1 = self.selected_features[feature_idx]
                    feat2 = self.selected_features[feature_idx + 1]
                    
                    new_feature_name = f"{feat1}_{feat2}_interaction"
                    self.combined_train[new_feature_name] = (
                        self.combined_train[feat1] * self.combined_train[feat2]
                    )
                    engineered_features.append(new_feature_name)
                else:
                    break
            except Exception as e:
                logger.warning(f"Could not create interaction feature: {e}")
                break
        
        self.engineered_features = engineered_features
        
        # Store feature engineering info
        self.preprocessing_summary['feature_engineering'] = {
            'engineered_features': engineered_features,
            'engineering_methods': [
                'Claim to Premium Ratio',
                'Age Risk Score',
                'Vehicle Value to Claim Ratio', 
                'Claim Complexity Score',
                'Time-based Risk Score'
            ],
            'total_engineered': len(engineered_features)
        }
        
        logger.info(f"Feature engineering completed. Created {len(engineered_features)} new features: {engineered_features}")
        return engineered_features
