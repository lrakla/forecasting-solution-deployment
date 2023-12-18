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

Repository structure
