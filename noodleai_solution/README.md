Time-Series-Sales-Forecasting
==============================

To get started - 
1. Clone this repository
3. Run the following commands sequentially
```bash
cd noodleai_solution
docker build -t noodleai_solution .
docker run -d -it noodleai_solution bash
docker ps # to indentify container id
docker exec -it <container-id> bash
python main.py
```
Optionally, you can give --store <store-number> --dept <dept-number> to get forecasts for a particular date
```python
python main.py --store 1 --dept 2
```
The final write-up is in [noodleai_solution/reports/results.pdf](https://github.com/lrakla/noodleai-take-home/blob/2e2badbd6de4782ebab04bbad92f0578f6787baa/noodleai_solution/reports/results.pdf)

Repository structure
```
└── noodleai_solution/
    ├── data/
    │   ├── encoded/
    │   │   └── encoded.csv
    │   ├── forecasts/
    │   │   └── forecasts.csv
    │   ├── processed/
    │   │   └── processed.csv
    │   └── raw/
    │       └── data.csv
    ├── notebooks/
    │   ├── eda1.ipynb
    │   ├── eda2.ipynb
    │   ├── hyperparameter.ipynb
    │   └── deepar.py
    ├── report/
    │   ├── figures/
    │   └── results.ipynb
    │   └── results.pdf
    ├── src/
    │   ├── data/
    │   │   ├── process_dataset.py
    │   │   └── split_dataset.py
    │   ├── features/
    │   │   └── create_features.py
    │   ├── models/
    │   │   └── ml_models.py
    │   ├── visualizations/
    │   │   └── visualize.py
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── CONFIG.yaml
    │   └── models_config.yaml
    ├── requirements.txt
    ├── Dockerfile
    ├── main.py
    └── README.md
```
