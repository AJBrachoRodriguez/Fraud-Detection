# **Fraud detection**  📘
![database with api](img/frauddetection.png)

### **Description**  🗒️

Fraud detection is one of the main critical nodes in the financial world. The dataset used here comes from a financial insitution in Africa. The aim
of the project is to establish if a transaction is fraudulent or not using advanced machine learning algorithms such as Logistic Regression.


### **Table of Contents**  📑

- [Content of the repository](#content-of-the-repository)
- [How to Install and Run the Project](#how-to-install-and-run-the-project)
- [How to use the project](#how-to-use-the-project)
- [Contributions](#Contributions)
- [Credits](#credits)
- [Licence](#Licence)

### **Content of the repository**  🔡

1. notebook.ipynb
2. requirements.txt
3. .gitignore
4. README.md
5. tuningDT.py
6. tuningLR.py
7. tuningNB.py
8. tuningRF.py

### **How to Install and Run the Project**  🏃

1. You must set up an python environment installing the package included in the requirements file (.txt). Then, you can use any Integrated Development Environment (IDE) such as Jupyter or Visual Studio Code to run the
notebook.ipynb file.

2. You need to have installed Apache Spark in your PC, you can follow this guide is you have Linux/Ubuntu (otherwise, you´ll need to find another guide):

https://medium.com/@alexangelb/how-to-install-pyspark-in-linux-ubuntu-2f81a4006a36

3. Use the following command in you terminal (in the directory where the python scripts are) one by one:

$spark-submit --driver-memory 10g tuningDT.py
$spark-submit --driver-memory 10g tuningLR.py
$spark-submit --driver-memory 10g tuningNB.py
$spark-submit --driver-memory 10g tuningRF.py

4. Then, some folders with models "tuned" will appear in your currect directory:

/tuningDT/
/tuningLR/
/tuningNB/
/tuningRF/

### **How to use the project**  📂

You need to download the csv file from here:

https://www.kaggle.com/datasets/ealaxi/paysim1/download

Then, start deploying the project following the steps explained in the previous stage.

### **Status of the project**  🚉

The project was built in collaboration with Victor Carracedo and Elvis Donayre. The optimization in the PySpark implementation is currently being deployed.

### **Contributions**  ✍️

We would like you to encourage to contribute in any form to the project through this public repository.

### **Licence**  👮

*MIT* Licence
