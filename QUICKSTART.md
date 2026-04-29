# Quick Start Guide - Bias & Fairness Analyzer

## 🚀 Quick Installation & Running

### Step 1: Install Dependencies

Open PowerShell or Command Prompt in the project directory and run:

```bash
pip install -r requirements.txt
```

**Note:** This will download ~1GB of packages including PyTorch and transformers. The first run will also download BERT models (~400MB for English, ~500MB for Arabic).

### Step 2: Run the Application

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at http://localhost:8501

## 📖 Quick Tutorial

### Mode 1: Single Text Analysis (Fastest!)

1. Select language: English or Arabic
2. Choose "Single Text Analysis" mode
3. Type or paste your text
4. Click "Analyze Text"
5. View bias scores and detailed analysis

**Example texts to try:**

**English:**
```
The doctor asked the nurse to help him with the procedure.
```

**Arabic:**
```
طلب الطبيب من الممرضة أن تساعده في الإجراء.
```

### Mode 2: Batch Dataset Analysis

1. Choose "Batch Dataset Analysis"
2. Click "Run Batch Analysis"
3. Wait for analysis to complete (~30 seconds)
4. View fairness scores and download results

### Mode 3: WEAT Analysis

1. Choose "WEAT Analysis"
2. Click "Run WEAT Analysis"
3. See effect size and statistical significance

### Mode 4: Full Benchmark

1. Choose "Full Benchmark"
2. Click "Run Full Benchmark"
3. Wait 1-2 minutes for complete evaluation
4. Download detailed report

## 🐛 Troubleshooting

### Issue: "No module named 'transformers'"
**Solution:** Run `pip install -r requirements.txt` again

### Issue: "CUDA out of memory"
**Solution:** This is normal - the code automatically uses CPU if GPU is not available

### Issue: Models downloading slowly
**Solution:** First run requires internet to download models. Be patient (~5-10 minutes)

### Issue: streamlit command not found
**Solution:** 
```bash
python -m streamlit run app.py
```

## 📊 Understanding the Results

### Bias Scores
- **0.0 - 0.2**: Low bias (Fair)
- **0.2 - 0.5**: Moderate bias
- **0.5 - 1.0**: High bias (Biased)

### Fairness Scores
- **90-100**: Excellent (Grade A)
- **80-89**: Good (Grade B)
- **70-79**: Fair (Grade C)
- **60-69**: Poor (Grade D)
- **Below 60**: Very Poor (Grade F)

### Gender Bias Direction
- **male-biased**: More male references
- **female-biased**: More female references
- **neutral**: Balanced or no gender references

## 💡 Tips for Best Results

1. **For accurate results:** Use complete sentences
2. **For Arabic:** Make sure text is properly encoded (UTF-8)
3. **For batch analysis:** Use the provided datasets first
4. **For WEAT:** Results show implicit biases in the model itself

## 📁 Output Files

Results are saved in the `results/` directory:
- **benchmark_*.json**: Raw benchmark data
- **report_*.txt**: Human-readable reports
- **Downloaded CSV**: From the dashboard

## 🎓 For Your Graduation Presentation

### Key Features to Demonstrate:

1. **Bilingual Support**: Switch between English and Arabic
2. **Real-time Analysis**: Type text and get instant results
3. **Comprehensive Metrics**: WEAT, StereoSet, custom fairness scores
4. **Visualization**: Charts showing bias breakdown
5. **Evaluation**: Compare filtered vs unfiltered outputs

### Talking Points:

- "This system fills a gap by providing an interactive tool for bias detection"
- "Supports both English and Arabic - important for regional context"
- "Uses BERT, state-of-the-art transformer model"
- "Implements industry-standard metrics like WEAT from academic research"
- "Can process batches for large-scale analysis"

## 📞 Need Help?

Check the main [README.md](README.md) for detailed documentation.

---

**Created for:** Graduation Project 2026  
**Author:** Abubaker S. Issa  
**Last Updated:** January 2026
