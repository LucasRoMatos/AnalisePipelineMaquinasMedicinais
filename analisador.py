import pandas as pd  
import numpy as np  
from sklearn.ensemble import IsolationForest  
import matplotlib.pyplot as plt  
import seaborn as sns  
from datetime import datetime, timedelta  
import random  

def create_raw_data(num_entries=100):  

    '''  
    Gera dados de log de acesso aleatórios para equipamentos médicos.  
    Args:  
        num_entries (int): Número de entradas de log a serem geradas  
    Returns:  
        list: Lista de dicionários contendo os registros de acesso simulados  
    '''  

    equipment_ids = [f"E{str(i).zfill(3)}" for i in range(1, 21)]  
    access_types = ["read", "write", "execute", "admin", "config"]   
    raw_data = []  
    base_time = datetime.now() - timedelta(days=7)  

    for i in range(num_entries):  

        time_offset = random.randint(0, 60*24*7)  
        timestamp = base_time + timedelta(minutes=time_offset)  

        equipment_id = random.choices(  
            equipment_ids,  
            weights=[10 if i < 5 else 2 for i in range(len(equipment_ids))],  
            k=1  
        )[0]

        access_type = random.choices(  
            access_types,  
            weights=[50, 30, 15, 3, 2],  
            k=1  

        )[0]  

        if access_type == "read":  
            status = "success" if random.random() < 0.98 else "failure"  
        elif access_type == "write":  
            status = "success" if random.random() < 0.95 else "failure"  
        elif access_type == "execute":  
            status = "success" if random.random() < 0.92 else "failure"  
        else:  
            status = "success" if random.random() < 0.85 else "failure"  

        if random.random() < 0.02:  
            if random.random() < 0.5:  
                for _ in range(random.randint(5, 8)):  
                    raw_data.append({  
                        "timestamp": timestamp + timedelta(seconds=random.uniform(0, 1)),  
                        "equipment_id": equipment_id,  
                        "access_type": random.choice(["write", "execute", "admin"]),  
                        "status": "failure" if random.random() < 0.8 else "success"  
                    })  
                    timestamp += timedelta(seconds=random.uniform(0, 1))  

            else:  
                timestamp = timestamp.replace(hour=random.randint(0, 5))  
                raw_data.append({  
                    "timestamp": timestamp,  
                    "equipment_id": equipment_id,  
                    "access_type": random.choice(["admin", "config"]),  
                    "status": "failure" if random.random() < 0.7 else "success"  
                })  

        raw_data.append({  
            "timestamp": timestamp,  
            "equipment_id": equipment_id,  
            "access_type": access_type,  
            "status": status  
        })  

    return raw_data  

def process_raw_data(raw_data):  

    '''  
    Converte os dados brutos em um DataFrame pandas.  
    Args:  
        raw_data (list): Lista de dicionários com os dados brutos  
    Returns:  
        DataFrame: DataFrame pandas com os dados estruturados  
    '''  

    log_df = pd.DataFrame(raw_data)  
    return log_df  

def ingest_logs(log_df):  

    '''  
    Realiza a ingestão dos dados, convertendo tipos e formatando campos. 
    Args:  
        log_df (DataFrame): DataFrame com os dados brutos 
    Returns:  
        DataFrame: DataFrame com os dados processados  

    '''  
    log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])  

    return log_df  

  

def preprocess_logs(log_df):  

    '''  
    Pré-processa os dados extraindo features adicionais para análise.  
    Args:  

        log_df (DataFrame): DataFrame com os dados ingeridos  
    Returns:  
        DataFrame: DataFrame com features adicionais  
    '''  

    log_df["access_count"] = log_df.groupby("equipment_id")["equipment_id"].transform("count")  
    log_df["hour_of_day"] = log_df["timestamp"].dt.hour  
    log_df["is_failure"] = (log_df["status"] == "failure").astype(int)  
    log_df["is_sensitive"] = (log_df["access_type"].isin(["admin", "config"])).astype(int)  
    log_df["is_night"] = ((log_df["hour_of_day"] >= 0) & (log_df["hour_of_day"] <= 5)).astype(int)  

    return log_df  

  

def detect_anomalies(log_df):  

    ''' 
    Detecta acessos anômalos usando o algoritmo Isolation Forest.  
    Args:  
        log_df (DataFrame): DataFrame com os dados pré-processados  
    Returns:  
        DataFrame: DataFrame com scores de anomalia adicionados  
    '''  

    model = IsolationForest(contamination=0.05, random_state=42)  
    features = ["access_count", "hour_of_day", "is_failure", "is_sensitive", "is_night"] 
    model.fit(log_df[features])  
    log_df["anomaly_score"] = model.decision_function(log_df[features])  

    return log_df  

  

def classify_access(log_df):  

    '''
    Classifica os acessos em Normal, Suspeito ou Crítico baseado nos scores.  
    Args:  
        log_df (DataFrame): DataFrame com os scores de anomalia  
    Returns:  
        DataFrame: DataFrame com a coluna de categoria de acesso   
    ''' 

    critical_threshold = np.percentile(log_df["anomaly_score"], 5)  
    log_df["access_category"] = np.where(  
        log_df["anomaly_score"] <= critical_threshold,  
        "Crítico",  
        np.where(  
            (log_df["is_failure"] == 1) | (log_df["is_sensitive"] == 1),  
            "Suspeito",  
            "Normal"  
        )  
    )  

    return log_df  

def generate_alerts(log_df):  

    '''  
    Gera alertas para acessos classificados como críticos
    Args:  
        log_df (DataFrame): DataFrame com os acessos classificados  
    '''  

    critical_access = log_df[log_df["access_category"] == "Crítico"].sort_values("anomaly_score")  
    if not critical_access.empty:  
        print("\n=== ALERTAS DE ACESSOS CRÍTICOS ===")  
        print(f"Total de acessos críticos detectados: {len(critical_access)}\n")  

        for _, row in critical_access.iterrows():  
            print(f"[{row['timestamp']}] ALERTA: Acesso Crítico ao equipamento {row['equipment_id']}")  
            print(f"   Tipo: {row['access_type']}, Status: {row['status']}")  
            print(f"   Horário: {row['hour_of_day']}h, Score: {row['anomaly_score']:.2f}")  
            print("-" * 50)  

    else:  
        print("Nenhum acesso crítico detectado.")  

def visualize_logs(log_df):  

    '''  
    Gera visualizações dos dados processados e classificados.  
    Args:  
        log_df (DataFrame): DataFrame com os dados completos  
    ''' 

    plt.figure(figsize=(14, 10))  

    # Gráfico 1: Distribuição de categorias  

    plt.subplot(2, 2, 1)  
    category_counts = log_df["access_category"].value_counts()  
    colors = {"Normal": "green", "Suspeito": "orange", "Crítico": "red"}  
    plt.pie(
        category_counts, labels=category_counts.index, autopct='%1.1f%%',   
        colors=[colors[x] for x in category_counts.index]
        )  
    plt.title("Distribuição de Categorias de Acesso")  

    # Gráfico 2: Acessos por hora do dia  

    plt.subplot(2, 2, 2)  
    for category in ["Normal", "Suspeito", "Crítico"]:  
        subset = log_df[log_df["access_category"] == category]  
        sns.kdeplot(subset["hour_of_day"], label=category, color=colors[category])  

    plt.title("Distribuição de Acessos por Hora do Dia")  
    plt.xlabel("Hora do Dia")  
    plt.ylabel("Densidade")  
    plt.legend()  

    # Gráfico 3: Dispersão de acessos  

    plt.subplot(2, 1, 2)  
    sns.scatterplot(  
        x="timestamp", y="equipment_id", hue="access_category",  
        data=log_df, palette=colors,  
        size="is_sensitive", sizes=(50, 200), alpha=0.7  
    )  

    plt.title("Acessos aos Equipamentos ao Longo do Tempo")  
    plt.xlabel("Horário")  
    plt.ylabel("ID do Equipamento")  
    plt.legend(title="Categoria")  

    plt.tight_layout()  
    plt.show()  

  

if __name__ == "__main__":  

    '''  
    Pipeline principal que executa todas as etapas do processo:  

    1. Geração de dados  
    2. Processamento  
    3. Análise  
    4. Visualização  

    '''  

    print("Gerando dados aleatórios com poucos erros...")  
    raw_data = create_raw_data(300)  

    print("Processando dados brutos...")  
    log_df = process_raw_data(raw_data)  
    log_df = ingest_logs(log_df)  
    log_df = preprocess_logs(log_df) 

    print("Analisando padrões de acesso...")  
    log_df = detect_anomalies(log_df)  
    log_df = classify_access(log_df)  
    generate_alerts(log_df)  

    print("\nGerando visualizações...")  
    visualize_logs(log_df)  

    print("\n=== RESUMO ESTATÍSTICO ===")  
    print("Total de acessos:", len(log_df))  
    print("\nDistribuição de status:")  
    print(log_df["status"].value_counts())  
    print("\nDistribuição de categorias:")  
    print(log_df["access_category"].value_counts()) 