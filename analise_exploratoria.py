import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

INPUT_PATH = Path("dados_brasileirao/transformados")
FIGURES_PATH = Path("figuras_analise")
INSIGHTS_PATH = Path("dados_brasileirao/insights")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cards = pd.read_csv(INPUT_PATH / "base_transformada_cartoes.csv")
    matches = pd.read_csv(INPUT_PATH / "base_transformada_partidas.csv")
    
    agg_visao_geral = pd.read_csv(INPUT_PATH / "agg_visao_geral.csv")
    agg_cartoes_mes = pd.read_csv(INPUT_PATH / "agg_cartoes_por_ano_mes.csv")
    agg_mandante_visitante = pd.read_csv(INPUT_PATH / "agg_cartoes_mandante_visitante.csv")
    agg_formacao = pd.read_csv(INPUT_PATH / "agg_cartoes_por_formacao.csv")
    agg_posicao = pd.read_csv(INPUT_PATH / "agg_cartoes_por_posicao.csv")
    agg_minuto = pd.read_csv(INPUT_PATH / "agg_cartoes_por_faixa_minuto.csv")
    agg_eficiencia = pd.read_csv(INPUT_PATH / "agg_eficiencia_disciplina.csv")
    
    aggregations = {
        "visao_geral": agg_visao_geral,
        "cartoes_mes": agg_cartoes_mes,
        "mandante_visitante": agg_mandante_visitante,
        "formacao": agg_formacao,
        "posicao": agg_posicao,
        "minuto": agg_minuto,
        "eficiencia": agg_eficiencia,
    }
    
    return cards, matches, aggregations


def setup_paths():
    FIGURES_PATH.mkdir(exist_ok=True)
    INSIGHTS_PATH.mkdir(exist_ok=True)


def analise_fator_casa(matches: pd.DataFrame) -> dict:
    insight = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fator Casa: Mando de Campo e Comportamento Tático", fontsize=14, fontweight="bold")
    
    mandante_media = matches["cartoes_mandante"].mean()
    visitante_media = matches["cartoes_visitante"].mean()
    
    insight["media_cartoes_mandante"] = round(mandante_media, 2)
    insight["media_cartoes_visitante"] = round(visitante_media, 2)
    insight["diferenca_percentual"] = round((mandante_media - visitante_media) / visitante_media * 100, 2)
    
    ax = axes[0, 0]
    dados = pd.DataFrame({
        "Posicao": ["Mandante", "Visitante"],
        "Media_Cartoes": [mandante_media, visitante_media]
    })
    ax.bar(dados["Posicao"], dados["Media_Cartoes"], color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Media de Cartoes por Partida")
    ax.set_title("Media de Cartoes: Mandante vs Visitante")
    for i, v in enumerate(dados["Media_Cartoes"]):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontweight="bold")
    
    ax = axes[0, 1]
    resultado_cartoes = matches.groupby("resultado_mandante")[["cartoes_mandante", "cartoes_visitante"]].mean()
    resultado_cartoes.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Media de Cartoes")
    ax.set_title("Cartoes por Resultado do Mandante")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(["Mandante", "Visitante"])
    
    ax = axes[1, 0]
    interestadual = matches.groupby("jogo_interestadual")[["cartoes_mandante", "cartoes_visitante"]].mean()
    interestadual_labels = ["Mesmo Estado", "Estados Diferentes"]
    interestadual.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Media de Cartoes")
    ax.set_title("Influencia de Jogo Interestadual")
    ax.set_xticklabels(interestadual_labels, rotation=45)
    ax.legend(["Mandante", "Visitante"])
    
    ax = axes[1, 1]
    volume = matches["volume_cartoes"].value_counts().sort_index()
    ax.bar(volume.index, volume.values, color="#2ca02c")
    ax.set_ylabel("Quantidade de Partidas")
    ax.set_title("Distribuicao de Partidas por Volume de Cartoes")
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "01_fator_casa.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return insight


def analise_tatica_disciplina(matches: pd.DataFrame, agg_formacao: pd.DataFrame) -> dict:
    insight = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Tatica vs Disciplina: Formacao e Taxa de Cartoes", fontsize=14, fontweight="bold")
    
    formacao_cartoes = matches.groupby("perfil_tatico_mandante")["cartoes_mandante"].agg(["mean", "std", "count"])
    insight["formacao_stats"] = formacao_cartoes.to_dict()
    
    ax = axes[0, 0]
    formacao_cartoes["mean"].plot(kind="bar", ax=ax, color="#d62728", yerr=formacao_cartoes["std"])
    ax.set_ylabel("Media de Cartoes por Partida")
    ax.set_title("Cartoes por Perfil Tatico (Mandante)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    ax = axes[0, 1]
    formacao_sample = agg_formacao.nlargest(10, "cartoes_por_partida_time")
    ax.barh(range(len(formacao_sample)), formacao_sample["cartoes_por_partida_time"], color="#9467bd")
    ax.set_yticks(range(len(formacao_sample)))
    ax.set_yticklabels(formacao_sample["formacao_clube"] + " (" + formacao_sample["perfil_tatico_clube"] + ")")
    ax.set_xlabel("Cartoes por Partida")
    ax.set_title("Top 10 Formacoes Mais Punidas")
    
    ax = axes[1, 0]
    formacao_count = matches.groupby("perfil_tatico_mandante").size()
    ax.pie(formacao_count.values, labels=formacao_count.index, autopct="%1.1f%%", colors=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Distribuicao de Formacoes Taticas")
    
    ax = axes[1, 1]
    formacao_vitoria = matches.groupby("perfil_tatico_mandante")["resultado_mandante"].value_counts().unstack(fill_value=0)
    formacao_vitoria.plot(kind="bar", ax=ax, stacked=False)
    ax.set_ylabel("Quantidade de Partidas")
    ax.set_title("Resultado por Formacao Tatica")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(title="Resultado")
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "02_tatica_disciplina.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return insight


def analise_sindrome_visitante(matches: pd.DataFrame, cards: pd.DataFrame) -> dict:
    insight = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sindrome do Visitante: Comportamento Fora de Casa", fontsize=14, fontweight="bold")
    
    cartoes_local = cards.groupby("local_clube")["cartao"].value_counts().unstack(fill_value=0)
    insight["cartoes_por_local"] = cartoes_local.to_dict()
    
    ax = axes[0, 0]
    cartoes_media = cards.groupby("local_clube")["cartao_peso"].mean()
    ax.bar(cartoes_media.index, cartoes_media.values, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Peso Medio de Cartao")
    ax.set_title("Gravidade Media: Mandante vs Visitante")
    for i, v in enumerate(cartoes_media.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    
    ax = axes[0, 1]
    comparacao = cards.groupby(["local_clube", "jogo_interestadual"]).size().unstack(fill_value=0)
    comparacao.plot(kind="bar", ax=ax, color=["#2ca02c", "#d62728"])
    ax.set_ylabel("Quantidade de Cartoes")
    ax.set_title("Cartoes por Local e Tipo de Jogo")
    ax.set_xticklabels(["Mandante", "Visitante"], rotation=0)
    ax.legend(["Mesmo Estado", "Estados Diferentes"])
    
    ax = axes[1, 0]
    visitante_punido = matches["visitante_mais_punido"].value_counts()
    ax.pie([visitante_punido[False], visitante_punido[True]], 
           labels=["Mandante Mais Punido", "Visitante Mais Punido"],
           autopct="%1.1f%%", colors=["#1f77b4", "#ff7f0e"])
    ax.set_title("Proporcao: Quem e Mais Punido?")
    
    ax = axes[1, 1]
    minuto_local = cards.groupby("local_clube")["minuto_total"].mean()
    ax.bar(minuto_local.index, minuto_local.values, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Minuto Medio do Cartao")
    ax.set_title("Minuto Medio: Quando Visitantes Levam Cartao?")
    for i, v in enumerate(minuto_local.values):
        ax.text(i, v + 1, f"{v:.0f}min", ha="center", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "03_sindrome_visitante.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return insight


def analise_risco_posicao(cards: pd.DataFrame, matches: pd.DataFrame, agg_posicao: pd.DataFrame) -> dict:
    insight = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Risco por Posicao: Quem Recebe Mais Cartao?", fontsize=14, fontweight="bold")
    
    posicao_cartoes = cards.groupby("posicao")["cartao"].value_counts().unstack(fill_value=0)
    insight["cartoes_por_posicao"] = posicao_cartoes.to_dict()
    
    ax = axes[0, 0]
    total_por_posicao = cards["posicao"].value_counts()
    ax.barh(total_por_posicao.index, total_por_posicao.values, color="#1f77b4")
    ax.set_xlabel("Quantidade de Cartoes")
    ax.set_title("Total de Cartoes por Posicao")
    
    ax = axes[0, 1]
    peso_medio = cards.groupby("posicao")["cartao_peso"].mean().sort_values(ascending=False)
    ax.barh(peso_medio.index, peso_medio.values, color="#d62728")
    ax.set_xlabel("Peso Medio de Cartao (1=Amarelo, 2=Vermelho)")
    ax.set_title("Gravidade Media: Qual Posicao Leva Vermelhos?")
    
    ax = axes[1, 0]
    taxa_cartoes = cards.groupby("posicao").size() / matches.shape[0]
    ax.barh(taxa_cartoes.index, taxa_cartoes.values, color="#2ca02c")
    ax.set_xlabel("Taxa de Cartoes por Partida")
    ax.set_title("Risco Relativo por Posicao")
    
    ax = axes[1, 1]
    amarelo_vermelho = cards.groupby("posicao")[["cartao_amarelo", "cartao_vermelho"]].sum()
    amarelo_vermelho.plot(kind="barh", ax=ax, stacked=True, color=["#ffdd57", "#ff5252"])
    ax.set_xlabel("Quantidade de Cartoes")
    ax.set_title("Composicao: Amarelos vs Vermelhos por Posicao")
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "04_risco_posicao.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return insight


def analise_hora_desespero(cards: pd.DataFrame, agg_minuto: pd.DataFrame) -> dict:
    insight = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Hora do Desespero: Cartoes ao Longo do Jogo", fontsize=14, fontweight="bold")
    
    minuto_cartoes = cards.groupby("faixa_minuto")["cartao"].value_counts().unstack(fill_value=0)
    insight["cartoes_por_minuto"] = minuto_cartoes.to_dict()
    
    ax = axes[0, 0]
    faixa_total = cards["faixa_minuto"].value_counts().sort_index()
    ax.plot(range(len(faixa_total)), faixa_total.values, marker="o", linewidth=2, markersize=8, color="#1f77b4")
    ax.set_xticks(range(len(faixa_total)))
    ax.set_xticklabels(faixa_total.index, rotation=45)
    ax.set_ylabel("Quantidade de Cartoes")
    ax.set_title("Distribuicao de Cartoes por Faixa de Minuto")
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    etapa_cartoes = cards.groupby("etapa_jogo")["cartao"].value_counts().unstack(fill_value=0)
    etapa_cartoes.plot(kind="bar", ax=ax, color=["#ffdd57", "#ff5252"])
    ax.set_ylabel("Quantidade de Cartoes")
    ax.set_title("Cartoes por Etapa do Jogo")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(title="Tipo")
    
    ax = axes[1, 0]
    reta_final_pct = cards.groupby("posicao")["cartao_reta_final"].mean() * 100
    ax.barh(reta_final_pct.index, reta_final_pct.values, color="#d62728")
    ax.set_xlabel("Percentual de Cartoes na Reta Final (%)")
    ax.set_title("Proporacao: Cartoes na Reta Final (75+ min)")
    
    ax = axes[1, 1]
    minuto_real = cards.groupby("minuto_total")["cartao"].count()
    ax.plot(minuto_real.index, minuto_real.values, linewidth=1, color="#2ca02c", alpha=0.7)
    ax.fill_between(minuto_real.index, minuto_real.values, alpha=0.3, color="#2ca02c")
    ax.set_xlabel("Minuto da Partida")
    ax.set_ylabel("Quantidade de Cartoes")
    ax.set_title("Distribuicao Detalhada por Minuto Real")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "05_hora_desespero.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return insight


def analise_bater_ganha_jogo(matches: pd.DataFrame) -> dict:
    insight = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Bater Ganha Jogo? Disciplina vs Resultado", fontsize=14, fontweight="bold")
    
    ax = axes[0, 0]
    resultado_cartoes = matches.groupby("resultado_mandante")[["cartoes_mandante", "cartoes_visitante"]].mean()
    resultado_cartoes.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Media de Cartoes")
    ax.set_title("Cartoes Recebidos vs Resultado do Mandante")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(["Mandante", "Visitante"])
    
    ax = axes[0, 1]
    volume_resultado = matches.groupby(["volume_cartoes", "resultado_mandante"]).size().unstack(fill_value=0)
    volume_resultado.plot(kind="bar", ax=ax, stacked=False)
    ax.set_ylabel("Quantidade de Partidas")
    ax.set_title("Resultado por Volume de Cartoes")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(title="Resultado")
    
    ax = axes[1, 0]
    menos_cartoes = matches.groupby("time_com_menos_cartoes").size()
    ax.pie(menos_cartoes.values, labels=menos_cartoes.index, autopct="%1.1f%%", 
           colors=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Quem Ganhou com Menos Cartoes?")
    
    ax = axes[1, 1]
    gols_cartoes = matches.groupby("total_gols_partida")["cartoes_total"].mean()
    ax.plot(gols_cartoes.index, gols_cartoes.values, marker="s", linewidth=2, markersize=6, color="#d62728")
    ax.set_xlabel("Total de Gols na Partida")
    ax.set_ylabel("Media de Cartoes")
    ax.set_title("Relacao: Gols vs Cartoes")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "06_bater_ganha_jogo.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    vitoria_menos_cartoes = (matches[matches["resultado_mandante"] == "Vitoria"]["time_com_menos_cartoes"] == "Mandante").sum()
    vitoria_total = (matches["resultado_mandante"] == "Vitoria").sum()
    insight["proporcao_vitoria_menos_cartoes"] = round(vitoria_menos_cartoes / vitoria_total * 100, 2)
    
    return insight


def compilar_relatorio(insights: dict) -> pd.DataFrame:
    relatorio = pd.DataFrame([
        {
            "analise": "Fator Casa",
            "metrica": "Media Cartoes Mandante",
            "valor": insights["fator_casa"]["media_cartoes_mandante"]
        },
        {
            "analise": "Fator Casa",
            "metrica": "Media Cartoes Visitante",
            "valor": insights["fator_casa"]["media_cartoes_visitante"]
        },
        {
            "analise": "Fator Casa",
            "metrica": "Diferenca Percentual",
            "valor": f"{insights['fator_casa']['diferenca_percentual']}%"
        },
    ])
    
    relatorio.to_csv(INSIGHTS_PATH / "relatorio_insights.csv", index=False)
    return relatorio


def main() -> None:
    setup_paths()
    
    print("Carregando dados transformados...")
    cards, matches, aggregations = load_data()
    
    print("Executando analises exploratoria...")
    
    insights = {}
    
    print("  - Fator Casa...")
    insights["fator_casa"] = analise_fator_casa(matches)
    
    print("  - Tatica vs Disciplina...")
    insights["tatica_disciplina"] = analise_tatica_disciplina(matches, aggregations["formacao"])
    
    print("  - Sindrome do Visitante...")
    insights["sindrome_visitante"] = analise_sindrome_visitante(matches, cards)
    
    print("  - Risco por Posicao...")
    insights["risco_posicao"] = analise_risco_posicao(cards, matches, aggregations["posicao"])
    
    print("  - Hora do Desespero...")
    insights["hora_desespero"] = analise_hora_desespero(cards, aggregations["minuto"])
    
    print("  - Bater Ganha Jogo...")
    insights["bater_ganha"] = analise_bater_ganha_jogo(matches)
    
    print("Compilando relatorio...")
    compilar_relatorio(insights)
    
    print(f"\nFiguras geradas em: {FIGURES_PATH}")
    print(f"Relatorio salvo em: {INSIGHTS_PATH}")
    print("\nAnalise exploratoria concluida com sucesso!")


if __name__ == "__main__":
    main()
