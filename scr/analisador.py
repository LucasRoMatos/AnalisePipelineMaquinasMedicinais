import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import random
import re

def criar_dados_brutos(num_entradas=100):
    '''
    Gera dados aleatórios de log de acesso para equipamentos médicos.
    Produz dados estruturados e mensagens de log semiestruturadas.
    '''
    ids_equipamentos = [f"E{str(i).zfill(3)}" for i in range(1, 21)]
    tipos_acesso = ["leitura", "escrita", "execução", "admin", "config"] 
    dados_brutos = []
    data_base = datetime.now() - timedelta(days=7)

    for i in range(num_entradas):
        if random.random() < 0.5:
            # Logs estruturados
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
            # Mensagens de log semiestruturadas
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
    '''
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
    Processa dados brutos, lidando com logs estruturados e semiestruturados.
    '''
    logs_estruturados = []
    
    for entrada in dados_brutos:
        if "log_cru" in entrada:
            analisado = analisar_com_regex(entrada)
            if analisado:
                logs_estruturados.append(analisado)
        else:
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

def exportar_para_powerbi(df_log, nome_arquivo='dados_logs_powerbi.xlsx'):
    """
    Exporta os dados processados para um arquivo que pode ser consumido pelo Power BI.
    """
    cols_export = [
        'timestamp', 'equipment_id', 'access_type', 'status',
        'contagem_acessos', 'hora_do_dia', 'eh_falha', 'eh_sensivel', 'eh_noite',
        'score_anomalia', 'categoria_acesso'
    ]
    
    df_export = df_log[cols_export].copy()
    df_export['timestamp'] = df_export['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_export.to_excel(nome_arquivo, index=False)
    print(f"\nDados exportados para {nome_arquivo} - Prontos para importação no Power BI")

def criar_datasets_analiticos(df_log):
    """
    Cria datasets resumidos e agregados para facilitar a criação de dashboards no Power BI.
    """
    # Dataset 1: Estatísticas por equipamento
    stats_equipamento = df_log.groupby('equipment_id').agg({
        'timestamp': ['count', 'min', 'max'],
        'eh_falha': 'sum',
        'score_anomalia': 'mean',
        'categoria_acesso': lambda x: (x == 'Crítico').sum()
    }).reset_index()
    stats_equipamento.columns = ['equipment_id', 'total_acessos', 'primeiro_acesso', 'ultimo_acesso', 
                               'total_falhas', 'score_anomalia_medio', 'acessos_criticos']
    
    # Dataset 2: Estatísticas por hora do dia
    stats_horario = df_log.groupby('hora_do_dia').agg({
        'equipment_id': 'count',
        'eh_falha': 'sum',
        'score_anomalia': 'mean',
        'categoria_acesso': lambda x: (x == 'Crítico').sum()
    }).reset_index()
    stats_horario.columns = ['hora_do_dia', 'total_acessos', 'total_falhas', 
                            'score_anomalia_medio', 'acessos_criticos']
    
    # Dataset 3: Estatísticas por tipo de acesso
    stats_tipo_acesso = df_log.groupby('access_type').agg({
        'equipment_id': 'count',
        'eh_falha': 'sum',
        'score_anomalia': 'mean',
        'categoria_acesso': lambda x: (x == 'Crítico').sum()
    }).reset_index()
    stats_tipo_acesso.columns = ['access_type', 'total_acessos', 'total_falhas', 
                               'score_anomalia_medio', 'acessos_criticos']
    
    # Exporta todos os datasets
    with pd.ExcelWriter('dados_analiticos_powerbi.xlsx') as writer:
        stats_equipamento.to_excel(writer, sheet_name='Por_Equipamento', index=False)
        stats_horario.to_excel(writer, sheet_name='Por_Horario', index=False)
        stats_tipo_acesso.to_excel(writer, sheet_name='Por_Tipo_Acesso', index=False)
    
    print("Datasets analíticos exportados para 'dados_analiticos_powerbi.xlsx'")

def visualizar_logs(df_log):
    """
    Substitui as visualizações do matplotlib por exportação para Power BI.
    """
    print("\nPreparando dados para visualização no Power BI...")
    exportar_para_powerbi(df_log)
    criar_datasets_analiticos(df_log)
    
    print("\nInstruções para importar no Power BI:")
    print("1. Abra o Power BI Desktop")
    print("2. Vá em 'Obter Dados' > 'Excel'")
    print("3. Selecione um dos arquivos gerados:")
    print("   - dados_logs_powerbi.xlsx (dados completos)")
    print("   - dados_analiticos_powerbi.xlsx (dados agregados)")
    print("4. Crie suas visualizações usando os campos disponíveis")

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