{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "private_outputs": true,
      "provenance": [],
      "authorship_tag": "ABX9TyNjBSJdNTXlxYbBPFkk5nAP",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/shrawanbishnoi395-dot/chatbot/blob/main/Project1\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Setup and Data** **Cleaning**"
      ],
      "metadata": {
        "id": "RrC7pJ17tcRs"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lus70oHItVV6"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import datetime as dt\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import classification_report\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 1. Load Dataset\n",
        "df = pd.read_csv('/content/data.csv (1).zip', encoding='ISO-8859-1')\n",
        "\n"
      ],
      "metadata": {
        "id": "cGnBHbOfu22y"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 2. Data Cleaning\n",
        "df = df.dropna(subset=['CustomerID']) # Remove rows without CustomerID\n",
        "df = df[df['Quantity'] > 0]           # Remove returns/cancellations\n",
        "df['TotalSum'] = df['Quantity'] * df['UnitPrice']\n",
        "df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])\n",
        "\n",
        "print(\"Data Cleaned. Total Rows:\", len(df))"
      ],
      "metadata": {
        "id": "xq0gvJO7u7qP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**RFM Analysis (Segmentation)**"
      ],
      "metadata": {
        "id": "suL8NZlBtbDu"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 1. Calculate Recency, Frequency, Monetary\n",
        "snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)\n",
        "rfm = df.groupby('CustomerID').agg({\n",
        "    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,\n",
        "    'InvoiceNo': 'count',\n",
        "    'TotalSum': 'sum'\n",
        "})\n",
        "\n",
        "rfm.rename(columns={\n",
        "    'InvoiceDate': 'Recency',\n",
        "    'InvoiceNo': 'Frequency',\n",
        "    'TotalSum': 'MonetaryValue'\n",
        "}, inplace=True)\n",
        "\n"
      ],
      "metadata": {
        "id": "PZPh0esCvP9_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 2. Assign Scores (1-5)\n",
        "rfm['R'] = pd.qcut(rfm['Recency'], q=5, labels=range(5, 0, -1))\n",
        "rfm['F'] = pd.qcut(rfm['Frequency'], q=5, labels=range(1, 6))\n",
        "rfm['M'] = pd.qcut(rfm['MonetaryValue'], q=5, labels=range(1, 6))\n",
        "rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis=1)\n",
        "\n",
        "# 3. Define Segments\n",
        "def segment_me(df):\n",
        "    if df['RFM_Score'] >= 13: return 'Champions'\n",
        "    elif df['RFM_Score'] >= 9: return 'Loyal'\n",
        "    elif df['RFM_Score'] >= 5: return 'At Risk'\n",
        "    else: return 'Lost'\n",
        "\n",
        "rfm['General_Segment'] = rfm.apply(segment_me, axis=1)\n",
        "print(rfm['General_Segment'].value_counts())"
      ],
      "metadata": {
        "id": "M3yazbR8vh5H"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Cohort Analysis (Retention Heatmap)**"
      ],
      "metadata": {
        "id": "sFp2AsBpvt5O"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 1. Create Cohort Month\n",
        "df['InvoiceMonth'] = df['InvoiceDate'].apply(lambda x: dt.datetime(x.year, x.month, 1))\n",
        "df['CohortMonth'] = df.groupby('CustomerID')['InvoiceMonth'].transform('min')\n",
        "\n",
        "# 2. Calculate Cohort Index\n",
        "def get_date_int(df, column):\n",
        "    year = df[column].dt.year\n",
        "    month = df[column].dt.month\n",
        "    return year, month\n",
        "\n",
        "inv_y, inv_m = get_date_int(df, 'InvoiceMonth')\n",
        "coh_y, coh_m = get_date_int(df, 'CohortMonth')\n",
        "df['CohortIndex'] = (inv_y - coh_y) * 12 + (inv_m - coh_m) + 1\n",
        "\n",
        "# 3. Build Retention Matrix\n",
        "cohort_counts = df.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().unstack()\n",
        "retention = cohort_counts.divide(cohort_counts.iloc[:, 0], axis=0)\n",
        "\n",
        "# 4. Plot Heatmap\n",
        "plt.figure(figsize=(12, 8))\n",
        "sns.heatmap(retention, annot=True, fmt='.0%', cmap='BuGn')\n",
        "plt.title('Customer Retention Rates')\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "ukRQGA5LvxKP"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
