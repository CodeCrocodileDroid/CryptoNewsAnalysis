import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import ast
import re
import warnings

warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# Load and preprocess the data
def load_and_preprocess_data(filename):
    """Load CSV and preprocess sentiment data"""
    print("Loading data...")
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} rows")

    # Fix date parsing with multiple formats
    def parse_date(date_str):
        try:
            # Try multiple date formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M',
                '%Y-%m-%d',
                '%m/%d/%Y %H:%M:%S',
                '%m/%d/%Y %H:%M',
                '%m/%d/%Y'
            ]

            for fmt in formats:
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except:
                    continue

            # If none of the formats work, use pandas built-in parser
            return pd.to_datetime(date_str, errors='coerce')
        except:
            return pd.NaT

    df['date'] = df['date'].apply(parse_date)

    # Drop rows with invalid dates
    initial_count = len(df)
    df = df.dropna(subset=['date'])
    print(f"Dropped {initial_count - len(df)} rows with invalid dates")

    # Parse sentiment dictionary from string to actual dictionary
    def parse_sentiment(sentiment_str):
        try:
            # Clean the string and parse as dictionary
            sentiment_str = str(sentiment_str).replace("'", "\"")
            # Handle NaN or None values
            if pd.isna(sentiment_str) or sentiment_str == 'nan':
                return {'class': 'neutral', 'polarity': 0.0, 'subjectivity': 0.0}
            sentiment_dict = ast.literal_eval(sentiment_str)
            return sentiment_dict
        except Exception as e:
            # Default values if parsing fails
            return {'class': 'neutral', 'polarity': 0.0, 'subjectivity': 0.0}

    # Apply parsing
    print("Parsing sentiment data...")
    df['sentiment_dict'] = df['sentiment'].apply(parse_sentiment)

    # Extract sentiment features
    df['sentiment_class'] = df['sentiment_dict'].apply(lambda x: x.get('class', 'neutral'))
    df['sentiment_polarity'] = df['sentiment_dict'].apply(lambda x: x.get('polarity', 0.0))
    df['sentiment_subjectivity'] = df['sentiment_dict'].apply(lambda x: x.get('subjectivity', 0.0))

    # Extract time features
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.day_name()
    df['date_only'] = df['date'].dt.date
    df['month'] = df['date'].dt.month_name()

    print(f"Data preprocessing complete. Final dataset: {len(df)} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return df


# 1. BASIC DATA ANALYSIS AND VISUALIZATION
def create_basic_visualizations(df):
    """Create basic exploratory data visualizations"""
    print("Creating basic visualizations...")

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cryptocurrency News Analysis Dashboard', fontsize=16, fontweight='bold')

    try:
        # 1. Sentiment Distribution Pie Chart
        sentiment_counts = df['sentiment_class'].value_counts()
        colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}
        sentiment_colors = [colors.get(s, '#95a5a6') for s in sentiment_counts.index]

        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index,
                       autopct='%1.1f%%', colors=sentiment_colors, startangle=90)
        axes[0, 0].set_title('Sentiment Distribution', fontweight='bold')

        # 2. Subject Distribution
        subject_counts = df['subject'].value_counts().head(10)
        axes[0, 1].barh(subject_counts.index, subject_counts.values, color='#3498db')
        axes[0, 1].set_title('Top 10 News Subjects', fontweight='bold')
        axes[0, 1].set_xlabel('Number of Articles')

        # 3. Source Distribution
        source_counts = df['source'].value_counts().head(5)
        axes[0, 2].bar(source_counts.index, source_counts.values, color='#9b59b6')
        axes[0, 2].set_title('Top 5 News Sources', fontweight='bold')
        axes[0, 2].set_ylabel('Number of Articles')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # 4. Sentiment Polarity Over Time
        daily_sentiment = df.groupby('date_only')['sentiment_polarity'].mean()
        axes[1, 0].plot(daily_sentiment.index, daily_sentiment.values,
                        marker='o', color='#e74c3c', linewidth=2, markersize=4)
        axes[1, 0].set_title('Daily Average Sentiment Polarity', fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Average Polarity')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 5. Hourly Sentiment Analysis
        hourly_sentiment = df.groupby('hour')['sentiment_polarity'].mean()
        axes[1, 1].bar(hourly_sentiment.index, hourly_sentiment.values, color='#f39c12')
        axes[1, 1].set_title('Hourly Average Sentiment', fontweight='bold')
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Average Polarity')
        axes[1, 1].set_xticks(range(0, 24, 3))

        # 6. Subjectivity vs Polarity Scatter
        scatter = axes[1, 2].scatter(df['sentiment_subjectivity'], df['sentiment_polarity'],
                                     c=df['sentiment_polarity'], cmap='RdYlGn',
                                     alpha=0.6, s=50)
        axes[1, 2].set_title('Subjectivity vs Polarity', fontweight='bold')
        axes[1, 2].set_xlabel('Subjectivity')
        axes[1, 2].set_ylabel('Polarity')
        plt.colorbar(scatter, ax=axes[1, 2])

        plt.tight_layout()
        plt.savefig('basic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Basic visualizations created and saved as 'basic_analysis.png'")

    except Exception as e:
        print(f"Error in basic visualizations: {e}")
        # Create a simple plot if the detailed one fails
        plt.figure(figsize=(10, 6))
        df['sentiment_class'].value_counts().plot(kind='bar', color=['green', 'blue', 'red'])
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('basic_sentiment.png', dpi=300)
        plt.show()


# 2. ADVANCED VISUALIZATIONS
def create_advanced_visualizations(df):
    """Create more sophisticated visualizations"""
    print("Creating advanced visualizations...")

    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Heatmap: Sentiment by Subject and Source
        top_subjects = df['subject'].value_counts().head(8).index
        top_sources = df['source'].value_counts().head(5).index
        filtered_df = df[df['subject'].isin(top_subjects) & df['source'].isin(top_sources)]

        pivot_table = pd.pivot_table(filtered_df,
                                     values='sentiment_polarity',
                                     index='subject',
                                     columns='source',
                                     aggfunc='mean')

        sns.heatmap(pivot_table, cmap='RdYlGn', center=0,
                    ax=axes[0, 0], cbar_kws={'label': 'Average Polarity'},
                    annot=True, fmt='.2f')
        axes[0, 0].set_title('Heatmap: Average Sentiment by Subject and Source',
                             fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].tick_params(axis='y', rotation=0)

        # 2. Violin Plot: Sentiment Distribution by Subject
        top_subjects_violin = df['subject'].value_counts().head(5).index
        filtered_df_violin = df[df['subject'].isin(top_subjects_violin)]
        sns.violinplot(x='subject', y='sentiment_polarity', data=filtered_df_violin,
                       ax=axes[0, 1], palette='Set2')
        axes[0, 1].set_title('Sentiment Distribution by Top Subjects',
                             fontweight='bold')
        axes[0, 1].set_xlabel('Subject')
        axes[0, 1].set_ylabel('Sentiment Polarity')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Stacked Bar: Sentiment Composition by Source
        sentiment_by_source = pd.crosstab(df['source'], df['sentiment_class'],
                                          normalize='index')
        sentiment_by_source.plot(kind='bar', stacked=True,
                                 color=['#e74c3c', '#3498db', '#2ecc71'],
                                 ax=axes[1, 0])
        axes[1, 0].set_title('Sentiment Composition by News Source',
                             fontweight='bold')
        axes[1, 0].set_xlabel('Source')
        axes[1, 0].set_ylabel('Proportion')
        axes[1, 0].legend(title='Sentiment')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Time Series with Rolling Average
        daily_stats = df.groupby('date_only').agg({
            'sentiment_polarity': 'mean',
            'sentiment_subjectivity': 'mean',
            'title': 'count'
        }).rename(columns={'title': 'article_count'})

        # Calculate 3-day rolling averages
        daily_stats['polarity_rolling'] = daily_stats['sentiment_polarity'].rolling(3, min_periods=1).mean()

        axes[1, 1].plot(daily_stats.index, daily_stats['polarity_rolling'],
                        linewidth=2, color='red', label='3-day MA Polarity')
        axes[1, 1].bar(daily_stats.index, daily_stats['article_count'],
                       alpha=0.3, color='blue', label='Article Count')
        axes[1, 1].set_title('Sentiment Trends with Article Volume',
                             fontweight='bold')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Average Polarity')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('advanced_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Advanced visualizations created and saved as 'advanced_analysis.png'")

    except Exception as e:
        print(f"Error in advanced visualizations: {e}")


# 3. MACHINE LEARNING: SENTIMENT PREDICTION
def perform_sentiment_prediction(df):
    """Predict sentiment using machine learning"""
    print("\nPerforming sentiment prediction with machine learning...")

    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        # Prepare data for classification
        df_ml = df.copy()

        # Create binary sentiment labels (positive vs negative/neutral)
        df_ml['sentiment_binary'] = df_ml['sentiment_class'].apply(
            lambda x: 1 if x == 'positive' else 0
        )

        # Create text features from titles
        print("Creating text features...")
        vectorizer = CountVectorizer(max_features=100,
                                     stop_words='english',
                                     ngram_range=(1, 2))
        X_text = vectorizer.fit_transform(df_ml['title'].fillna('')).toarray()

        # Combine with numerical features
        X_numerical = df_ml[['sentiment_polarity', 'sentiment_subjectivity']].values
        X_combined = np.hstack([X_text, X_numerical])
        y = df_ml['sentiment_binary'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")

        # Train Random Forest classifier
        print("Training Random Forest classifier...")
        rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        rf.fit(X_train, y_train)

        # Make predictions
        y_pred = rf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print("\n" + "=" * 50)
        print("MACHINE LEARNING RESULTS")
        print("=" * 50)
        print(f"\nModel Accuracy: {accuracy:.2%}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['Negative/Neutral', 'Positive']))

        # Feature importance
        feature_names = list(vectorizer.get_feature_names_out()) + ['polarity', 'subjectivity']
        importances = rf.feature_importances_
        top_features_idx = importances.argsort()[-15:][::-1]

        print("\nTop 15 Most Important Features for Sentiment Prediction:")
        for idx in top_features_idx:
            print(f"  {feature_names[idx]}: {importances[idx]:.4f}")

        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative/Neutral', 'Positive'],
                    yticklabels=['Negative/Neutral', 'Positive'])
        plt.title(f'Confusion Matrix - Sentiment Prediction (Accuracy: {accuracy:.2%})',
                  fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('sentiment_prediction.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Sentiment prediction visualization saved as 'sentiment_prediction.png'")

        return rf, vectorizer, accuracy

    except Exception as e:
        print(f"Error in sentiment prediction: {e}")
        print("Trying simplified approach...")

        # Simplified approach
        try:
            # Simple visualization of sentiment patterns
            plt.figure(figsize=(10, 6))

            # Group by subject and calculate average sentiment
            subject_sentiment = df.groupby('subject')['sentiment_polarity'].mean().sort_values()
            subject_sentiment.tail(10).plot(kind='barh', color='green', alpha=0.7, label='Most Positive')
            subject_sentiment.head(10).plot(kind='barh', color='red', alpha=0.7, label='Most Negative')

            plt.title('Sentiment by Subject (Top/Bottom 10)', fontweight='bold')
            plt.xlabel('Average Sentiment Polarity')
            plt.ylabel('Subject')
            plt.legend()
            plt.tight_layout()
            plt.savefig('sentiment_by_subject.png', dpi=300)
            plt.show()

            return None, None, 0

        except Exception as e2:
            print(f"Simplified approach also failed: {e2}")
            return None, None, 0


# 4. TOPIC MODELING (Simplified)
def perform_topic_modeling_simple(df):
    """Simple topic analysis without LDA"""
    print("\nPerforming topic analysis...")

    try:
        # Extract common words from titles
        all_titles = ' '.join(df['title'].fillna('').astype(str).tolist()).lower()
        words = re.findall(r'\b[a-z]{4,}\b', all_titles)

        # Remove common stopwords
        stopwords = {'bitcoin', 'crypto', 'cryptocurrency', 'news', 'price', 'market',
                     'says', 'will', 'could', 'would', 'new', 'year', 'like', 'first',
                     'ethereum', 'token', 'tokens', 'also', 'may', 'one', 'time'}
        words = [w for w in words if w not in stopwords]

        from collections import Counter
        word_counts = Counter(words)

        # Get top 20 words
        top_words = word_counts.most_common(20)

        print("\nTop 20 Most Frequent Words in News Titles:")
        for word, count in top_words:
            print(f"  {word}: {count}")

        # Visualize top words
        plt.figure(figsize=(12, 6))
        words_list = [w[0] for w in top_words]
        counts_list = [w[1] for w in top_words]

        colors = plt.cm.viridis(np.linspace(0, 1, len(words_list)))
        plt.barh(words_list, counts_list, color=colors)
        plt.title('Top 20 Words in Cryptocurrency News Titles', fontweight='bold')
        plt.xlabel('Frequency')
        plt.tight_layout()
        plt.savefig('topic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Topic analysis visualization saved as 'topic_analysis.png'")

        return top_words

    except Exception as e:
        print(f"Error in topic modeling: {e}")
        return []


# 5. CLUSTERING ANALYSIS
def perform_clustering_analysis(df):
    """Cluster articles based on sentiment"""
    print("\nPerforming clustering analysis...")

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Prepare features for clustering
        features = df[['sentiment_polarity', 'sentiment_subjectivity']].copy()

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Use KMeans with 3 clusters
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(features_scaled)

        # Visualize clusters
        plt.figure(figsize=(10, 8))

        # Define cluster names based on characteristics
        cluster_names = {
            0: 'Neutral/Objective',
            1: 'Positive/Subjective',
            2: 'Negative/Subjective'
        }

        colors = ['blue', 'green', 'red']

        for cluster_num in range(3):
            cluster_data = df[df['cluster'] == cluster_num]
            plt.scatter(cluster_data['sentiment_subjectivity'],
                        cluster_data['sentiment_polarity'],
                        c=colors[cluster_num],
                        label=cluster_names[cluster_num],
                        alpha=0.6, s=50)

        # Add centroids
        centroids = kmeans.cluster_centers_
        centroids_original = scaler.inverse_transform(centroids)
        plt.scatter(centroids_original[:, 1], centroids_original[:, 0],
                    c='black', s=200, alpha=0.8, marker='X', label='Centroids')

        plt.xlabel('Subjectivity')
        plt.ylabel('Sentiment Polarity')
        plt.title('Article Clusters Based on Sentiment and Subjectivity', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add cluster statistics
        for cluster_num in range(3):
            cluster_data = df[df['cluster'] == cluster_num]
            avg_polarity = cluster_data['sentiment_polarity'].mean()
            avg_subjectivity = cluster_data['sentiment_subjectivity'].mean()
            count = len(cluster_data)

            plt.annotate(
                f'Cluster {cluster_num}\n{count} articles\nPol: {avg_polarity:.2f}\nSub: {avg_subjectivity:.2f}',
                xy=(centroids_original[cluster_num, 1], centroids_original[cluster_num, 0]),
                xytext=(centroids_original[cluster_num, 1] + 0.05, centroids_original[cluster_num, 0] + 0.05),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=9)

        plt.tight_layout()
        plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Clustering analysis visualization saved as 'clustering_analysis.png'")

        # Print cluster statistics
        print("\nCluster Statistics:")
        for cluster_num in range(3):
            cluster_data = df[df['cluster'] == cluster_num]
            print(f"\nCluster {cluster_num} ({cluster_names[cluster_num]}):")
            print(f"  Number of articles: {len(cluster_data)}")
            print(f"  Avg polarity: {cluster_data['sentiment_polarity'].mean():.3f}")
            print(f"  Avg subjectivity: {cluster_data['sentiment_subjectivity'].mean():.3f}")

            # Most common subject in this cluster
            common_subject = cluster_data['subject'].mode()[0] if not cluster_data['subject'].mode().empty else 'N/A'
            print(f"  Most common subject: {common_subject}")

        return df, kmeans

    except Exception as e:
        print(f"Error in clustering analysis: {e}")
        return df, None


# 6. TIME SERIES ANALYSIS
def perform_time_series_analysis(df):
    """Analyze time patterns in sentiment"""
    print("\nPerforming time series analysis...")

    try:
        # Create daily statistics
        daily_stats = df.groupby('date_only').agg({
            'sentiment_polarity': 'mean',
            'title': 'count'
        }).rename(columns={'title': 'article_count'})

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Daily sentiment trend
        axes[0, 0].plot(daily_stats.index, daily_stats['sentiment_polarity'],
                        color='blue', linewidth=1, alpha=0.7)
        # Add 7-day moving average
        axes[0, 0].plot(daily_stats.index, daily_stats['sentiment_polarity'].rolling(7, min_periods=1).mean(),
                        color='red', linewidth=2, label='7-day MA')
        axes[0, 0].set_title('Daily Sentiment Trend', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Average Polarity')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Daily article volume
        axes[0, 1].bar(daily_stats.index, daily_stats['article_count'],
                       color='green', alpha=0.7)
        axes[0, 1].set_title('Daily Article Volume', fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Number of Articles')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Weekly pattern
        weekly_pattern = df.groupby('day_of_week')['sentiment_polarity'].mean()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern = weekly_pattern.reindex(day_order)

        axes[1, 0].bar(range(len(weekly_pattern)), weekly_pattern.values,
                       color=plt.cm.viridis(np.linspace(0, 1, len(weekly_pattern))))
        axes[1, 0].set_title('Average Sentiment by Day of Week', fontweight='bold')
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Average Polarity')
        axes[1, 0].set_xticks(range(len(weekly_pattern)))
        axes[1, 0].set_xticklabels([d[:3] for d in weekly_pattern.index], rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Hourly pattern
        hourly_pattern = df.groupby('hour')['sentiment_polarity'].mean()
        axes[1, 1].plot(hourly_pattern.index, hourly_pattern.values,
                        marker='o', linewidth=2, color='purple')
        axes[1, 1].fill_between(hourly_pattern.index, 0, hourly_pattern.values, alpha=0.3)
        axes[1, 1].set_title('Average Sentiment by Hour of Day', fontweight='bold')
        axes[1, 1].set_xlabel('Hour of Day (24h)')
        axes[1, 1].set_ylabel('Average Polarity')
        axes[1, 1].set_xticks(range(0, 24, 3))
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Time series analysis visualization saved as 'time_series_analysis.png'")

        # Calculate correlations
        correlation = daily_stats['sentiment_polarity'].corr(daily_stats['article_count'])
        print(f"\nCorrelation between daily article count and average sentiment: {correlation:.3f}")

        if correlation > 0.3:
            print("  Interpretation: More articles correlate with more positive sentiment")
        elif correlation < -0.3:
            print("  Interpretation: More articles correlate with more negative sentiment")
        else:
            print("  Interpretation: Weak correlation between article volume and sentiment")

        return correlation

    except Exception as e:
        print(f"Error in time series analysis: {e}")
        return 0


# 7. GENERATE INSIGHTS REPORT
def generate_insights_report(df, ml_accuracy=None):
    """Generate comprehensive insights from the analysis"""

    print("\n" + "=" * 60)
    print("COMPREHENSIVE INSIGHTS REPORT")
    print("=" * 60)

    insights = []

    # 1. Overall sentiment summary
    positive_pct = (df['sentiment_class'] == 'positive').mean() * 100
    negative_pct = (df['sentiment_class'] == 'negative').mean() * 100
    neutral_pct = (df['sentiment_class'] == 'neutral').mean() * 100

    insights.append(f"1. SENTIMENT OVERVIEW:")
    insights.append(f"   - Positive articles: {positive_pct:.1f}%")
    insights.append(f"   - Negative articles: {negative_pct:.1f}%")
    insights.append(f"   - Neutral articles: {neutral_pct:.1f}%")

    # 2. Most positive/negative subjects
    subject_sentiment = df.groupby('subject')['sentiment_polarity'].mean().sort_values()

    insights.append(f"\n2. SENTIMENT BY SUBJECT:")
    insights.append(f"   - Most positive subject: {subject_sentiment.index[-1]} ({subject_sentiment.iloc[-1]:.3f})")
    insights.append(f"   - Most negative subject: {subject_sentiment.index[0]} ({subject_sentiment.iloc[0]:.3f})")

    # 3. Source analysis
    source_sentiment = df.groupby('source')['sentiment_polarity'].mean().sort_values()

    insights.append(f"\n3. SOURCE ANALYSIS:")
    insights.append(f"   - Most positive source: {source_sentiment.index[-1]} ({source_sentiment.iloc[-1]:.3f})")
    insights.append(f"   - Most negative source: {source_sentiment.index[0]} ({source_sentiment.iloc[0]:.3f})")

    # 4. Time patterns
    hourly_sentiment = df.groupby('hour')['sentiment_polarity'].mean()
    best_hour = hourly_sentiment.idxmax()
    worst_hour = hourly_sentiment.idxmin()

    insights.append(f"\n4. TEMPORAL PATTERNS:")
    insights.append(f"   - Best sentiment hour: {best_hour}:00 ({hourly_sentiment.max():.3f})")
    insights.append(f"   - Worst sentiment hour: {worst_hour}:00 ({hourly_sentiment.min():.3f})")

    # 5. Volume analysis
    articles_per_day = df.groupby('date_only').size().mean()
    insights.append(f"\n5. VOLUME ANALYSIS:")
    insights.append(f"   - Average articles per day: {articles_per_day:.1f}")

    # 6. Machine Learning results
    if ml_accuracy is not None and ml_accuracy > 0:
        insights.append(f"\n6. MACHINE LEARNING RESULTS:")
        insights.append(f"   - Sentiment prediction accuracy: {ml_accuracy:.2%}")

    # 7. Key findings
    insights.append(f"\n7. KEY FINDINGS:")

    if positive_pct > 40:
        insights.append(f"   - Market sentiment is generally positive")
    elif negative_pct > 40:
        insights.append(f"   - Market sentiment is generally negative")
    else:
        insights.append(f"   - Market sentiment is mixed")

    # Check for subject patterns
    bitcoin_sentiment = df[df['subject'].str.contains('bitcoin', case=False, na=False)]['sentiment_polarity'].mean()
    if bitcoin_sentiment > 0.1:
        insights.append(f"   - Bitcoin-related news tends to be positive")
    elif bitcoin_sentiment < -0.1:
        insights.append(f"   - Bitcoin-related news tends to be negative")

    # Check source reliability
    source_volume = df['source'].value_counts()
    if len(source_volume) > 1:
        main_source = source_volume.index[0]
        insights.append(f"   - {main_source} is the most active news source")

    # Print all insights
    print("\n")
    for insight in insights:
        print(insight)

    # Save insights to file
    with open('crypto_news_insights.txt', 'w') as f:
        f.write("CRYPTOCURRENCY NEWS ANALYSIS INSIGHTS REPORT\n")
        f.write("=" * 50 + "\n\n")
        for insight in insights:
            f.write(insight + "\n")

    print(f"\n✓ Insights report saved to 'crypto_news_insights.txt'")


# 8. CREATE SUMMARY DASHBOARD
def create_summary_dashboard(df):
    """Create a one-page summary dashboard"""
    print("\nCreating summary dashboard...")

    try:
        fig = plt.figure(figsize=(16, 12))

        # Define grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Sentiment distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        sentiment_counts = df['sentiment_class'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index,
                autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Sentiment Distribution', fontweight='bold')

        # 2. Top subjects (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        subject_counts = df['subject'].value_counts().head(8)
        ax2.barh(subject_counts.index, subject_counts.values, color='#3498db')
        ax2.set_title('Top 8 Subjects', fontweight='bold')
        ax2.set_xlabel('Count')

        # 3. Daily sentiment (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        daily_sentiment = df.groupby('date_only')['sentiment_polarity'].mean()
        ax3.plot(daily_sentiment.index, daily_sentiment.values,
                 color='#e74c3c', linewidth=1, alpha=0.7)
        ax3.plot(daily_sentiment.index, daily_sentiment.values.rolling(7, min_periods=1).mean(),
                 color='red', linewidth=2)
        ax3.set_title('Daily Sentiment (7-day MA)', fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Polarity')
        ax3.tick_params(axis='x', rotation=45)

        # 4. Source distribution (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        source_counts = df['source'].value_counts().head(5)
        ax4.bar(source_counts.index, source_counts.values, color='#9b59b6')
        ax4.set_title('Top 5 Sources', fontweight='bold')
        ax4.set_ylabel('Count')
        ax4.tick_params(axis='x', rotation=45)

        # 5. Hourly pattern (middle)
        ax5 = fig.add_subplot(gs[1, 1])
        hourly_sentiment = df.groupby('hour')['sentiment_polarity'].mean()
        ax5.bar(hourly_sentiment.index, hourly_sentiment.values, color='#f39c12')
        ax5.set_title('Hourly Sentiment Pattern', fontweight='bold')
        ax5.set_xlabel('Hour of Day')
        ax5.set_ylabel('Avg Polarity')
        ax5.set_xticks(range(0, 24, 3))

        # 6. Scatter plot (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        scatter = ax6.scatter(df['sentiment_subjectivity'], df['sentiment_polarity'],
                              c=df['sentiment_polarity'], cmap='RdYlGn',
                              alpha=0.6, s=30)
        ax6.set_title('Subjectivity vs Polarity', fontweight='bold')
        ax6.set_xlabel('Subjectivity')
        ax6.set_ylabel('Polarity')
        plt.colorbar(scatter, ax=ax6)

        # 7. Weekly pattern (bottom left)
        ax7 = fig.add_subplot(gs[2, 0])
        weekly_pattern = df.groupby('day_of_week')['sentiment_polarity'].mean()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern = weekly_pattern.reindex(day_order)
        ax7.bar(range(len(weekly_pattern)), weekly_pattern.values,
                color=plt.cm.viridis(np.linspace(0, 1, len(weekly_pattern))))
        ax7.set_title('Sentiment by Day of Week', fontweight='bold')
        ax7.set_xlabel('Day')
        ax7.set_ylabel('Avg Polarity')
        ax7.set_xticks(range(len(weekly_pattern)))
        ax7.set_xticklabels([d[:3] for d in weekly_pattern.index], rotation=45)

        # 8. Text summary (bottom middle and right)
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('off')

        # Calculate summary statistics
        total_articles = len(df)
        date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        avg_polarity = df['sentiment_polarity'].mean()
        avg_subjectivity = df['sentiment_subjectivity'].mean()

        summary_text = f"""
        CRYPTO NEWS ANALYSIS SUMMARY

        Dataset Overview:
        • Total Articles: {total_articles:,}
        • Date Range: {date_range}
        • Unique Sources: {df['source'].nunique()}
        • Unique Subjects: {df['subject'].nunique()}

        Sentiment Metrics:
        • Avg Polarity: {avg_polarity:.3f}
        • Avg Subjectivity: {avg_subjectivity:.3f}
        • Positive: {positive_pct:.1f}%
        • Negative: {negative_pct:.1f}%
        • Neutral: {neutral_pct:.1f}%

        Key Insights:
        • Most Active Source: {df['source'].value_counts().index[0]}
        • Most Discussed Topic: {df['subject'].value_counts().index[0]}
        • Best Time for News: {best_hour}:00
        """

        ax8.text(0.05, 0.95, summary_text, fontsize=10,
                 verticalalignment='top', family='monospace')

        plt.suptitle('Cryptocurrency News Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig('summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Summary dashboard saved as 'summary_dashboard.png'")

    except Exception as e:
        print(f"Error creating summary dashboard: {e}")


# MAIN EXECUTION FUNCTION
def main():
    """Main function to run all analyses"""

    print("=" * 60)
    print("CRYPTOCURRENCY NEWS ANALYSIS TOOL")
    print("=" * 60)

    try:
        # Load and preprocess data
        df = load_and_preprocess_data('cryptonews.csv')

        if len(df) == 0:
            print("Error: No valid data found in the CSV file!")
            return

        # Create visualizations
        create_basic_visualizations(df)
        create_advanced_visualizations(df)

        # Machine Learning Analysis
        ml_model, vectorizer, ml_accuracy = perform_sentiment_prediction(df)

        # Topic Analysis
        top_words = perform_topic_modeling_simple(df)

        # Clustering Analysis
        df, kmeans_model = perform_clustering_analysis(df)

        # Time Series Analysis
        correlation = perform_time_series_analysis(df)

        # Create Summary Dashboard
        create_summary_dashboard(df)

        # Generate Insights Report
        generate_insights_report(df, ml_accuracy)

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("\nGenerated Output Files:")
        print("1. basic_analysis.png - Basic charts and graphs")
        print("2. advanced_analysis.png - Advanced visualizations")
        print("3. sentiment_prediction.png - ML prediction results")
        print("4. topic_analysis.png - Word frequency analysis")
        print("5. clustering_analysis.png - Article clustering")
        print("6. time_series_analysis.png - Temporal patterns")
        print("7. summary_dashboard.png - One-page summary")
        print("8. crypto_news_insights.txt - Detailed insights report")

    except Exception as e:
        print(f"\nCritical error in main analysis: {e}")
        print("\nTrying minimal analysis...")

        # Minimal fallback
        try:
            df = pd.read_csv('cryptonews.csv')
            print(f"Loaded {len(df)} rows")

            # Just show basic info
            plt.figure(figsize=(10, 6))
            if 'sentiment' in df.columns:
                # Simple sentiment count
                sentiment_counts = df['sentiment'].value_counts().head(10)
                sentiment_counts.plot(kind='bar')
                plt.title('Top 10 Sentiment Values')
                plt.xlabel('Sentiment')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig('simple_analysis.png', dpi=300)
                plt.show()

            print(f"Data columns: {df.columns.tolist()}")
            print(f"Data shape: {df.shape}")

        except Exception as e2:
            print(f"Minimal analysis also failed: {e2}")


# Run the analysis
if __name__ == "__main__":
    main()