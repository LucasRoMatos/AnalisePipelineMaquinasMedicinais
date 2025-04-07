import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import re  # Importação adicionada para expressões regulares

def criar_dados_brutos(num_entradas=100):
    '''
    Gera dados aleatórios de log de acesso para equipamentos médicos.
    Agora produz tanto dados estruturados quanto mensagens de log semiestruturadas.
    '''
    ids_equipamentos = [f"E{str(i).zfill(3)}" for i in range(1, 21)]
    tipos_acesso = ["leitura", "escrita", "execução", "admin", "config"] 
    dados_brutos = []
    data_base = datetime.now() - timedelta(days=7)

    for i in range(num_entradas):
        # Gera 50% de logs estruturados e 50% semiestruturados
        if random.random() < 0.5:
            # Logs estruturados (formato original)
            deslocamento_tempo = random.randint(0, 60*24*7)
            timestamp = data_base + timedelta(minutes=deslocamento_tempo)
            id_equipamento = random.choices(
                ids_equipamentos,
                weights=[10 if i < 5 else 2 for i in range(len(ids_equipamentos))],
                k=1
            )[0]

            tipo_acesso = random.choices(
                tipos_acesso,
                weights=[50, 30, 15, 3, 2],
                k=1
            )[0]

            if tipo_acesso == "leitura":
                status = "sucesso" if random.random() < 0.98 else "falha"
            elif tipo_acesso == "escrita":
                status = "sucesso" if random.random() < 0.95 else "falha"
            elif tipo_acesso == "execução":
                status = "sucesso" if random.random() < 0.92 else "falha"
            else:
                status = "sucesso" if random.random() < 0.85 else "falha"

            if random.random() < 0.02:
                if random.random() < 0.5:
                    for _ in range(random.randint(5, 8)):
                        dados_brutos.append({
                            "timestamp": timestamp + timedelta(seconds=random.uniform(0, 1)),
                            "equipment_id": id_equipamento,
                            "access_type": random.choice(["escrita", "execução", "admin"]),
                            "status": "falha" if random.random() < 0.8 else "sucesso"
                        })
                        timestamp += timedelta(seconds=random.uniform(0, 1))
                else:
                    timestamp = timestamp.replace(hour=random.randint(0, 5))
                    dados_brutos.append({
                        "timestamp": timestamp,
                        "equipment_id": id_equipamento,
                        "access_type": random.choice(["admin", "config"]),
                        "status": "falha" if random.random() < 0.7 else "sucesso"
                    })

            dados_brutos.append({
                "timestamp": timestamp,
                "equipment_id": id_equipamento,
                "access_type": tipo_acesso,
                "status": status
            })
        else:
            # Mensagens de log semiestruturadas para demonstração do parsing com regex
            deslocamento_tempo = random.randint(0, 60*24*7)
            timestamp = data_base + timedelta(minutes=deslocamento_tempo)
            id_equipamento = random.choices(
                ids_equipamentos,
                weights=[10 if i < 5 else 2 for i in range(len(ids_equipamentos))],
                k=1
            )[0]
            tipo_acesso = random.choices(
                tipos_acesso,
                weights=[50, 30, 15, 3, 2],
                k=1
            )[0]
            
            # Gera diferentes formatos de mensagens de log
            formatos_log = [
                f"ALERTA {timestamp.strftime('%Y-%m-%d %H:%M:%S')} {id_equipamento} {tipo_acesso} acesso do IP 192.168.1.{random.randint(1,100)}",
                f"ERRO {timestamp.strftime('%Y-%m-%d %H:%M:%S')} {id_equipamento} {tipo_acesso} falhou com código {random.randint(1000,9999)}",
                f"INFO {timestamp.strftime('%Y-%m-%d %H:%M:%S')} {id_equipamento} {tipo_acesso} concluído em {random.uniform(0.1, 2.0):.2f}s"
            ]
            dados_brutos.append({
                "log_cru": random.choice(formatos_log),
                "timestamp": timestamp
            })

    return dados_brutos

def analisar_com_regex(entrada_log):
    '''
    Analisa logs semiestruturados usando expressões regulares.
    Implementa uma gramática simples através de padrões regex.
    '''
    # Padrões regex para diferentes formatos de log (gramática implícita)
    padrao_alerta = r"ALERTA (?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<equipment_id>E\d{3}) (?P<access_type>\w+) acesso do IP (?P<ip>\d+\.\d+\.\d+\.\d+)"
    padrao_erro = r"ERRO (?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<equipment_id>E\d{3}) (?P<access_type>\w+) falhou com código (?P<codigo_erro>\d{4})"
    padrao_info = r"INFO (?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<equipment_id>E\d{3}) (?P<access_type>\w+) concluído em (?P<duracao>\d+\.\d+)s"
    
    for padrao in [padrao_alerta, padrao_erro, padrao_info]:
        correspondencia = re.match(padrao, entrada_log["log_cru"])
        if correspondencia:
            dados_analisados = correspondencia.groupdict()
            dados_analisados["timestamp"] = datetime.strptime(dados_analisados["timestamp"], '%Y-%m-%d %H:%M:%S')
            dados_analisados["status"] = "falha" if "ERRO" in entrada_log["log_cru"] else "sucesso"
            return dados_analisados
    
    return None

def processar_dados_brutos(dados_brutos):
    '''
    Processa dados brutos, agora lidando tanto com logs estruturados quanto semiestruturados.
    '''
    logs_estruturados = []
    
    for entrada in dados_brutos:
        if "log_cru" in entrada:
            # Processa logs semiestruturados com regex
            analisado = analisar_com_regex(entrada)
            if analisado:
                logs_estruturados.append(analisado)
        else:
            # Mantém logs já estruturados
            logs_estruturados.append(entrada)
    
    return pd.DataFrame(logs_estruturados)

def ingerir_logs(df_log):
    '''
    Ingere os dados convertendo tipos e formatando campos.
    '''
    df_log["timestamp"] = pd.to_datetime(df_log["timestamp"])
    return df_log

def preprocessar_logs(df_log):
    '''
    Pré-processa os dados extraindo features adicionais para análise.
    '''
    df_log["contagem_acessos"] = df_log.groupby("equipment_id")["equipment_id"].transform("count")
    df_log["hora_do_dia"] = df_log["timestamp"].dt.hour
    df_log["eh_falha"] = (df_log["status"] == "falha").astype(int)
    df_log["eh_sensivel"] = (df_log["access_type"].isin(["admin", "config"])).astype(int)
    df_log["eh_noite"] = ((df_log["hora_do_dia"] >= 0) & (df_log["hora_do_dia"] <= 5)).astype(int)
    return df_log

def detectar_anomalias(df_log):
    '''
    Detecta acessos anômalos usando o algoritmo Isolation Forest.
    '''
    modelo = IsolationForest(contamination=0.05, random_state=42)
    features = ["contagem_acessos", "hora_do_dia", "eh_falha", "eh_sensivel", "eh_noite"]
    modelo.fit(df_log[features])
    df_log["score_anomalia"] = modelo.decision_function(df_log[features])
    return df_log

def classificar_acessos(df_log):
    '''
    Classifica os acessos em Normal, Suspeito ou Crítico baseado nos scores.
    '''
    limite_critico = np.percentile(df_log["score_anomalia"], 5)
    df_log["categoria_acesso"] = np.where(
        df_log["score_anomalia"] <= limite_critico,
        "Crítico",
        np.where(
            (df_log["eh_falha"] == 1) | (df_log["eh_sensivel"] == 1),
            "Suspeito",
            "Normal"
        )
    )
    return df_log

def gerar_alertas(df_log):
    '''
    Gera alertas para acessos classificados como críticos.
    '''
    acessos_criticos = df_log[df_log["categoria_acesso"] == "Crítico"].sort_values("score_anomalia")
    if not acessos_criticos.empty:
        print("\n=== ALERTAS DE ACESSOS CRÍTICOS ===")
        print(f"Total de acessos críticos detectados: {len(acessos_criticos)}\n")

        for _, linha in acessos_criticos.iterrows():
            print(f"[{linha['timestamp']}] ALERTA: Acesso Crítico ao equipamento {linha['equipment_id']}")
            print(f"   Tipo: {linha['access_type']}, Status: {linha['status']}")
            print(f"   Horário: {linha['hora_do_dia']}h, Score: {linha['score_anomalia']:.2f}")
            print("-" * 50)
    else:
        print("Nenhum acesso crítico detectado.")

def visualizar_logs(df_log):
    '''
    Gera visualizações dos dados processados e classificados.
    '''
    plt.figure(figsize=(14, 10))

    # Gráfico 1: Distribuição de categorias
    plt.subplot(2, 2, 1)
    contagem_categorias = df_log["categoria_acesso"].value_counts()
    cores = {"Normal": "green", "Suspeito": "orange", "Crítico": "red"}
    plt.pie(
        contagem_categorias, labels=contagem_categorias.index, autopct='%1.1f%%',
        colors=[cores[x] for x in contagem_categorias.index]
    )
    plt.title("Distribuição de Categorias de Acesso")

    # Gráfico 2: Acessos por hora do dia
    plt.subplot(2, 2, 2)
    for categoria in ["Normal", "Suspeito", "Crítico"]:
        subset = df_log[df_log["categoria_acesso"] == categoria]
        sns.kdeplot(subset["hora_do_dia"], label=categoria, color=cores[categoria])

    plt.title("Distribuição de Acessos por Hora do Dia")
    plt.xlabel("Hora do Dia")
    plt.ylabel("Densidade")
    plt.legend()

    # Gráfico 3: Dispersão de acessos
    plt.subplot(2, 1, 2)
    sns.scatterplot(
        x="timestamp", y="equipment_id", hue="categoria_acesso",
        data=df_log, palette=cores,
        size="eh_sensivel", sizes=(50, 200), alpha=0.7
    )

    plt.title("Acessos aos Equipamentos ao Longo do Tempo")
    plt.xlabel("Horário")
    plt.ylabel("ID do Equipamento")
    plt.legend(title="Categoria")

    plt.tight_layout()
    plt.show()

def main():
    print("Gerando dados aleatórios com logs estruturados e semiestruturados...")
    dados_brutos = criar_dados_brutos(300)
    
    print("\nExemplo de logs semiestruturados gerados:")
    for i, log in enumerate([x for x in dados_brutos if "log_cru" in x][:3]):
        print(f"Exemplo {i+1}: {log['log_cru']}")
    
    print("\nProcessando dados brutos com parser regex...")
    df_log = processar_dados_brutos(dados_brutos)
    
    print("\nExemplo de dados analisados:")
    print(df_log.head())
    
    # Continuação do pipeline original
    df_log = ingerir_logs(df_log)
    df_log = preprocessar_logs(df_log)
    df_log = detectar_anomalias(df_log)
    df_log = classificar_acessos(df_log)
    gerar_alertas(df_log)
    visualizar_logs(df_log)
    
    print("\n=== RESUMO ESTATÍSTICO ===")
    print("Total de acessos:", len(df_log))
    print("\nDistribuição de status:")
    print(df_log["status"].value_counts())

if __name__ == "__main__":
    main()