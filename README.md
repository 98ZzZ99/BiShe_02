(1) Environment setup
1. Clone the repository: 
git clone <repository URL> && cd <repository>.
2. Create a virtual environment (recommended):
python -m venv venv
Activate on Unix/macOS
source venv/bin/activate
Activate on Windows PowerShell
venv\Scripts\Activate.ps1
3. Install dependencies: 
Run pip install -r requirements.txt. If a requirements file is not provided, install the core libraries manually (langgraph, langchain, pandas, numpy, matplotlib, plotly, openpyxl, sentence_transformers, faiss‑cpu, networkx, pyvis, neo4j, isotree, pyod, scikit‑learn, tensorflow, json‑repair, sktree). 
4. Configure the .env file (VERY IMPORTANT): 
Set your LLM API key (OPENAI_API_KEY or DASHSCOPE_API_KEY), DASHSCOPE_BASE_URL, OUTPUT_DIR and optional logging parameters. The program loads this file automatically at startup.
5. Prepare your data: 
Put CSV files in the data/ folder or use absolute/relative paths in your requests. If no path is supplied, the program defaults to data/hybrid_manufacturing_categorical.csv.

(2) Viewing interpreter in PyCharm
When running the project in PyCharm, verify that the correct interpreter is selected and all packages are installed:
1. Open the project in PyCharm.
2. Click IDE and Project Settings and select Settings (or press Ctrl+Alt+S).
3. In the left panel, select Project: <project name> → Python Interpreter. PyCharm displays the interpreter path and the list of installed packages. You can click Add Interpreter to switch to another environment. This ensures that the virtual environment contains all required packages.

(3) Running the application
Execute the main script:
python app_main.py
The program will log its progress and prompt for a request.

(4) Inspecting results
Console output: The application displays the top rows of the resulting DataFrame or any scalar result and indicates where files have been saved.
Generated files: Look in output/ for subdirectories named with timestamps. These contain CSV snapshots (e.g., qt_step_02_select_rows.csv), Excel reports (e.g., valve1_anomaly_results.xlsx) and images (PR, F1 and ROC curves).
Logs: Detailed logs are written to logs/app.log.

(5) Troubleshooting and notes
Missing dependencies: If the program complains about a missing package, install it in your active virtual environment using pip install <package>.
Invalid API key: If the intent router or summariser fails, check that your API key in .env is correct and that you have network access.
Data format requirements: The anomaly detection component expects the data to include a time_stamp column and, if available, an anomaly column for ground‑truth labels. Ensure your CSV files have these columns or let the preprocessing module correct column names.
Selecting algorithms: Include algorithm names such as EIF, AE, LOF, INNE, OCSVM or COPOD in your request to specify which anomaly detectors to run. If no algorithm is specified, the system runs all detectors and selects the one with the best PR-AUC or F1 score.
