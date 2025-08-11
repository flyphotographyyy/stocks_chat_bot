# Stock Signals PRO - Enhanced Multi-Source Analysis Platform

## Overview

Stock Signals PRO has been significantly enhanced into a comprehensive multi-source financial analysis platform. The application now provides advanced technical analysis, enhanced sentiment analysis from multiple news sources, interactive visualizations, and professional-grade market insights. 

Key upgrades include:
- **Multi-Source Data Integration**: Yahoo Finance, Google News, enhanced fundamental data
- **Advanced Technical Analysis**: 15+ indicators including Bollinger Bands, Stochastic, Williams %R, CCI, MFI
- **Enhanced Sentiment Analysis**: VADER + TextBlob with confidence scoring and trend analysis  
- **Interactive Charts**: Professional Plotly-based visualizations with candlestick charts
- **Comprehensive Scoring**: Multi-factor analysis with technical, momentum, volume, sentiment, and fundamental signals
- **Global Market Support**: Extended to 6 major markets including Japan and Australia
- **Professional UI**: Dashboard with metrics, color-coded signals, and downloadable results
- **Risk Management**: Enhanced risk profiles with detailed confidence scoring

The platform now serves as a professional-grade analysis tool suitable for serious traders and investors.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Framework**: Single-page application built with Streamlit for rapid development and deployment
- **Auto-refresh Capability**: Optional real-time updates using streamlit-autorefresh package with 30-minute intervals
- **Responsive UI**: Market status indicators and timezone-aware displays for multiple global markets

### Data Processing Architecture
- **Multi-source Data Integration**: Combines stock price data from Yahoo Finance with sentiment analysis from news feeds
- **Technical Analysis Engine**: Built-in calculation modules for various trading indicators and signals
- **Sentiment Analysis**: VADER sentiment analyzer for processing financial news and social media content
- **Market Calendar Integration**: Optional pandas-market-calendars for precise trading day validation

### Market Data Management
- **Global Market Support**: Timezone-aware handling for US, German, UK, and French markets
- **Real-time Market Status**: Dynamic market open/close detection with fallback mechanisms
- **Watchlist Persistence**: JSON-based local storage for user stock watchlists

### Error Handling and Fallbacks
- **Graceful Degradation**: Optional dependencies with fallback mechanisms when advanced features are unavailable
- **Exception Management**: Try-catch blocks for external API calls and optional package imports

## External Dependencies

### Core Data Sources
- **Yahoo Finance (yfinance)**: Primary source for real-time and historical stock price data
- **RSS News Feeds**: Financial news aggregation through feedparser for sentiment analysis

### Optional Market Data Services
- **pandas-market-calendars**: Precise trading calendar data for major global exchanges
- **External APIs**: Configurable integration points for additional financial data sources

### Analysis Libraries
- **VADER + TextBlob**: Dual-engine sentiment analysis with confidence scoring
- **NumPy/Pandas**: Mathematical computations and data manipulation
- **SciPy/Scikit-learn**: Statistical analysis and machine learning capabilities
- **Plotly**: Professional interactive charting and visualizations
- **BeautifulSoup**: Web scraping for additional data sources
- **Enhanced Technical Analysis**: 15+ professional indicators with multi-timeframe analysis

### Infrastructure Services
- **Streamlit Cloud**: Deployment platform for web application hosting
- **Local File System**: JSON-based persistence for user preferences and watchlists
- **Timezone Services (pytz)**: Global timezone conversion and market hours calculation