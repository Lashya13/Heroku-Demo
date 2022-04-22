import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json= {'Age':22,'Gender':0,'Internships':1,'CGPA':8,'Hostel':0,'HistoryOfBacklogs':0})

print(r.json())