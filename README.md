# OMR Checker Pro

A comprehensive NEET OMR (Optical Mark Recognition) sheet checking system with both traditional computer vision and AI-powered analysis capabilities.

## 🚀 Features

- **Dual Detection Methods**: Traditional CV + AI (Gemini/OpenRouter) analysis
- **Streamlit Web Interface**: User-friendly teacher and student panels
- **NEET Scoring Rules**: Accurate scoring with +4/-5/0 marking scheme
- **Answer Key Management**: Teachers can upload and save answer keys
- **Student Analysis**: Detailed performance reports with subject-wise breakdown
- **Cross-platform**: Works on Windows, macOS, and Linux

## 📋 Requirements

### System Requirements
- **Python**: 3.10 or higher
- **Node.js**: 16.0 or higher (for API server)
- **pnpm**: Latest version

### Python Dependencies
- Streamlit 1.28.1
- OpenCV (opencv-python-headless)
- NumPy, Pandas
- Pillow
- Google Generative AI
- python-dotenv

### API Keys (Optional but Recommended)
- **Gemini API Key**: For AI-powered OMR analysis
- **OpenRouter API Key**: Alternative AI provider

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/DEBNILPAL/Neet_omr_checker.git
cd Neet_omr_checker
```

### 2. Set up Python Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root:
```env
# AI Provider API Keys
GEMINI_API_KEY=your_gemini_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=google/gemini-flash-1.5

# API Server Configuration (if running API server)
PORT=3000
DATABASE_URL=postgres://user:password@localhost:5432/omr_checker
```

### 4. Install Node.js Dependencies (Optional - for API server)
```bash
# Install pnpm if not already installed
npm install -g pnpm

# Install Node.js dependencies
pnpm install
```

## 🚀 Running the Application

### Option 1: Streamlit App Only (Recommended)
```bash
# Navigate to the Streamlit app directory
cd artifacts/neet-omr-checker

# Run Streamlit using the virtual environment Python
# Windows:
..\..\.venv\Scripts\streamlit run app.py --server.port 5001
# macOS/Linux:
../../.venv/bin/streamlit run app.py --server.port 5001
```

### Option 2: Full Stack (Streamlit + API Server)
```bash
# Terminal 1: Start API server
cd artifacts/api-server
pnpm run dev

# Terminal 2: Start Streamlit app
cd artifacts/neet-omr-checker
# Windows:
..\..\.venv\Scripts\streamlit run app.py --server.port 5001
# macOS/Linux:
../../.venv/bin/streamlit run app.py --server.port 5001
```

## 📁 Project Structure

```
OMR-Checker-Pro/
├── artifacts/
│   ├── neet-omr-checker/          # Streamlit web application
│   │   ├── app.py                 # Main Streamlit app
│   │   ├── omr_processor.py       # CV-based OMR detection
│   │   ├── neet_scorer.py         # NEET scoring logic
│   │   ├── ai_analyzer.py         # AI-powered analysis
│   │   ├── requirements.txt       # Python dependencies
│   │   └── sample_omr.jpeg        # Sample OMR sheet
│   └── api-server/                # Node.js Express API server
│       ├── src/
│       ├── package.json
│       └── tsconfig.json
├── lib/                           # Shared libraries
├── .env.example                   # Environment variables template
├── requirements.txt               # Root Python dependencies
└── README.md                      # This file
```

## 🎯 How to Use

### For Teachers
1. **Access Teacher Panel**: Open the Streamlit app and navigate to "Teacher Panel"
2. **Upload Answer Key**: Upload a filled OMR sheet as the answer key
3. **Select Analysis Method**: Choose between Traditional CV, AI (Gemini), or AI (OpenRouter)
4. **Save Answer Key**: The system automatically saves the answer key for student evaluation
5. **Review Detection**: Check the detected answers and make corrections if needed

### For Students
1. **Access Student Panel**: Navigate to "Student Panel" in the Streamlit app
2. **Upload OMR Sheet**: Upload the student's answered OMR sheet
3. **Select Analysis Method**: Choose your preferred analysis method
4. **Get Results**: View detailed performance report with:
   - Total score and subject-wise marks
   - Question-wise analysis (correct/wrong/unattempted)
   - NEET-specific scoring breakdown
   - Optional questions handling

## 🔧 Configuration

### Streamlit Configuration
The Streamlit app is configured via `.streamlit/config.toml`:
```toml
[server]
port = 5000
headless = true
address = "0.0.0.0"

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### Detection Parameters
Traditional CV detection parameters in `omr_processor.py`:
- `filled_ratio`: Minimum fill ratio (0.08)
- `multi_similarity`: Multi-detection threshold (0.85)
- `top_margin`, `bottom_margin`: Vertical margins (5%, 3%)
- `left_margin`, `right_margin`: Horizontal margins (2%, 2%)

## 📊 NEET Scoring Rules

The system implements official NEET scoring:
- **Correct Answer**: +4 marks
- **Wrong Answer**: -5 marks
- **Unattempted**: 0 marks
- **Multiple Marked**: -5 marks (treated as wrong)

### Subject Structure
- **Physics**: Column 1 (Questions 1-50)
- **Chemistry**: Column 2 (Questions 1-50)
- **Botany**: Column 3 (Questions 1-50)
- **Zoology**: Column 4 (Questions 1-50)

### Question Types
- **Mandatory**: Questions 1-35 in each subject
- **Optional**: Questions 36-50 (only first 10 counted)

## 🤖 AI Analysis

### Gemini Integration
- Uses Google's Gemini AI for intelligent OMR analysis
- Supports vision-based analysis of OMR sheets
- Handles edge cases and ambiguous markings

### OpenRouter Integration
- Alternative AI provider with multiple model options
- Supports various models including Google Gemini via OpenRouter
- Fallback option when direct Gemini API is unavailable

## 🐛 Troubleshooting

### Common Issues

#### 1. "No module named 'google.generativeai'"
**Solution**: Ensure you're using the virtual environment:
```bash
# Windows
.venv\Scripts\activate
.venv\Scripts\streamlit run artifacts/neet-omr-checker/app.py

# macOS/Linux
source .venv/bin/activate
.venv/bin/streamlit run artifacts/neet-omr-checker/app.py
```

#### 2. "Gemini API error: 404 models/gemini-1.5-flash is not found"
**Solution**: Check your Gemini API key and ensure it has access to the model.

#### 3. Streamlit UI Issues
**Solution**: Ensure you're using Streamlit 1.28.1:
```bash
pip install streamlit==1.28.1
```

#### 4. Detection Accuracy Issues
**Solution**: 
- Ensure good quality scanned OMR sheets
- Check proper alignment and lighting
- Adjust detection parameters in `omr_processor.py` if needed

### Environment Verification
Check your setup with these commands:
```bash
# Check Python version
python --version

# Check virtual environment
which python  # macOS/Linux
where python  # Windows

# Check installed packages
pip list | grep streamlit
pip list | grep opencv
pip list | grep google-generativeai
```

## 📝 Development

### Running Tests
```bash
# Run Python tests (if available)
python -m pytest tests/

# Run Node.js tests (if available)
pnpm test
```

### Building for Production
```bash
# Build Node.js API server
cd artifacts/api-server
pnpm run build

# Streamlit app is ready to run as-is
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information about your problem

## 🔄 Updates

To update the project:
```bash
# Pull latest changes
git pull origin main

# Update Python dependencies
pip install -r requirements.txt

# Update Node.js dependencies
pnpm install
```

## 📊 Performance Tips

- Use high-quality scanned OMR sheets (300 DPI recommended)
- Ensure proper lighting and contrast in scanned images
- For batch processing, consider using the API server
- Monitor API usage when using AI analysis features

## 🔒 Security Notes

- Keep API keys secure and never commit them to version control
- Use environment variables for sensitive configuration
- Regularly update dependencies for security patches
- Consider using HTTPS in production deployments
