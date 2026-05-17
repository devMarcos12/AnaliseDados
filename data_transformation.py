import re
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("dados_brasileirao/base_integrada_projeto_limpa.csv")
OUTPUT_DIR = Path("dados_brasileirao/transformados")
CARD_OUTPUT = OUTPUT_DIR / "base_transformada_cartoes.csv"
MATCH_OUTPUT = OUTPUT_DIR / "base_transformada_partidas.csv"


def classify_formation(formation: str) -> str:
    if pd.isna(formation) or formation == "Não informado":
        return "Não informado"

    numbers = [int(value) for value in re.findall(r"\d+", formation)]
    if not numbers:
        return "Não informado"

    defenders = numbers[0]
    attackers = numbers[-1]

    if defenders >= 5:
        return "Conservadora"
    if attackers >= 3 or defenders <= 3:
        return "Ofensiva"
    return "Equilibrada"


def classify_score_margin(margin: int) -> str:
    margin = abs(margin)
    if margin == 0:
        return "Empate"
    if margin == 1:
        return "Vitoria apertada"
    if margin == 2:
        return "Vitoria moderada"
    return "Goleada"


def minmax_scale(series: pd.Series) -> pd.Series:
    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return pd.Series(0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def zscore_scale(series: pd.Series) -> pd.Series:
    std_value = series.std()
    if std_value == 0:
        return pd.Series(0, index=series.index)
    return (series - series.mean()) / std_value


def load_clean_dataset() -> pd.DataFrame:
    return pd.read_csv(INPUT_PATH, parse_dates=["data"])


def add_card_level_features(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df.copy()

    # enriquecimento temporal para analises de tendencia, sazonalidade e comportamento semanal
    transformed["trimestre"] = transformed["data"].dt.quarter
    transformed["dia_semana_num"] = transformed["data"].dt.dayofweek
    transformed["dia_semana"] = transformed["dia_semana_num"].map(
        {
            0: "Segunda-feira",
            1: "Terça-feira",
            2: "Quarta-feira",
            3: "Quinta-feira",
            4: "Sexta-feira",
            5: "Sábado",
            6: "Domingo",
        }
    )
    transformed["fim_de_semana"] = transformed["dia_semana_num"].isin([5, 6])

    transformed["faixa_rodada"] = pd.cut(
        transformed["rodada_partida"],
        bins=[0, 10, 20, 30, 38],
        labels=["Rodadas 1-10", "Rodadas 11-20", "Rodadas 21-30", "Rodadas 31-38"],
        include_lowest=True,
    )

    # enriquecimento analitico para comparar mandante, visitante e pressao de viagem
    transformed["local_clube"] = "Não identificado"
    transformed.loc[transformed["clube"].eq(transformed["mandante"]), "local_clube"] = "Mandante"
    transformed.loc[transformed["clube"].eq(transformed["visitante"]), "local_clube"] = "Visitante"
    transformed["jogo_interestadual"] = transformed["mandante_estado"].ne(transformed["visitante_estado"])

    transformed["formacao_clube"] = transformed["formacao_mandante"]
    transformed.loc[
        transformed["local_clube"].eq("Visitante"), "formacao_clube"
    ] = transformed["formacao_visitante"]
    transformed["perfil_tatico_clube"] = transformed["formacao_clube"].map(classify_formation)

    transformed["gols_clube"] = transformed["mandante_placar"]
    transformed["gols_adversario"] = transformed["visitante_placar"]
    transformed.loc[transformed["local_clube"].eq("Visitante"), "gols_clube"] = transformed[
        "visitante_placar"
    ]
    transformed.loc[transformed["local_clube"].eq("Visitante"), "gols_adversario"] = transformed[
        "mandante_placar"
    ]
    transformed["saldo_gols_clube"] = transformed["gols_clube"] - transformed["gols_adversario"]
    transformed["resultado_clube"] = "Empate"
    transformed.loc[transformed["saldo_gols_clube"] > 0, "resultado_clube"] = "Vitória"
    transformed.loc[transformed["saldo_gols_clube"] < 0, "resultado_clube"] = "Derrota"

    transformed["margem_placar"] = transformed["mandante_placar"] - transformed["visitante_placar"]
    transformed["tipo_placar"] = transformed["margem_placar"].map(classify_score_margin)
    transformed["total_gols_partida"] = transformed["mandante_placar"] + transformed["visitante_placar"]

    # segmentacao temporal do cartao para identificar desgaste e reta final do jogo
    transformed["faixa_minuto"] = pd.cut(
        transformed["minuto_total"],
        bins=[-1, 15, 30, 45, 60, 75, 90, 130],
        labels=["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"],
        include_lowest=True,
    )
    transformed["etapa_jogo"] = pd.cut(
        transformed["minuto_total"],
        bins=[-1, 45, 90, 130],
        labels=["1º tempo", "2º tempo", "Acréscimos"],
        include_lowest=True,
    )
    transformed["cartao_reta_final"] = transformed["minuto_total"].ge(75)
    transformed["cartao_acrescimos"] = transformed["minuto"].astype(str).str.contains(r"\+", regex=True)

    # transformacao categorica: ordinal para gravidade do cartao e dummies para filtros nominais
    transformed["cartao_peso"] = transformed["cartao"].map({"Amarelo": 1, "Vermelho": 2}).astype("Int64")
    transformed["cartao_amarelo"] = transformed["cartao"].eq("Amarelo").astype(int)
    transformed["cartao_vermelho"] = transformed["cartao"].eq("Vermelho").astype(int)
    transformed["eh_mandante"] = transformed["local_clube"].eq("Mandante").astype(int)
    transformed["eh_visitante"] = transformed["local_clube"].eq("Visitante").astype(int)

    position_dummies = pd.get_dummies(transformed["posicao"], prefix="posicao", dtype=int)
    transformed = pd.concat([transformed, position_dummies], axis=1)

    return transformed


def build_match_dataset(cards: pd.DataFrame) -> pd.DataFrame:
    match_columns = [
        "partida_id",
        "rodada_partida",
        "data",
        "ano",
        "mes",
        "trimestre",
        "dia_semana",
        "fim_de_semana",
        "faixa_rodada",
        "mandante",
        "visitante",
        "formacao_mandante",
        "formacao_visitante",
        "tecnico_mandante",
        "tecnico_visitante",
        "vencedor",
        "arena",
        "mandante_placar",
        "visitante_placar",
        "mandante_estado",
        "visitante_estado",
        "jogo_interestadual",
        "margem_placar",
        "tipo_placar",
        "total_gols_partida",
    ]
    matches = cards[match_columns].drop_duplicates("partida_id").copy()

    card_counts = (
        cards.pivot_table(
            index="partida_id",
            columns=["local_clube", "cartao"],
            values="atleta",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    card_counts.columns = [
        "_".join([str(part).lower() for part in column if str(part)])
        if isinstance(column, tuple)
        else column
        for column in card_counts.columns
    ]

    expected_columns = [
        "mandante_amarelo",
        "mandante_vermelho",
        "visitante_amarelo",
        "visitante_vermelho",
    ]
    for column in expected_columns:
        if column not in card_counts.columns:
            card_counts[column] = 0

    matches = matches.merge(card_counts, on="partida_id", how="left")
    matches[expected_columns] = matches[expected_columns].fillna(0).astype(int)

    matches["cartoes_mandante"] = matches["mandante_amarelo"] + matches["mandante_vermelho"]
    matches["cartoes_visitante"] = matches["visitante_amarelo"] + matches["visitante_vermelho"]
    matches["cartoes_total"] = matches["cartoes_mandante"] + matches["cartoes_visitante"]
    matches["diferenca_cartoes_visitante_mandante"] = (
        matches["cartoes_visitante"] - matches["cartoes_mandante"]
    )
    matches["visitante_mais_punido"] = matches["cartoes_visitante"].gt(matches["cartoes_mandante"])
    matches["time_com_menos_cartoes"] = "Empate em cartões"
    matches.loc[matches["cartoes_mandante"].lt(matches["cartoes_visitante"]), "time_com_menos_cartoes"] = "Mandante"
    matches.loc[matches["cartoes_visitante"].lt(matches["cartoes_mandante"]), "time_com_menos_cartoes"] = "Visitante"

    matches["resultado_mandante"] = "Empate"
    matches.loc[matches["mandante_placar"].gt(matches["visitante_placar"]), "resultado_mandante"] = "Vitória"
    matches.loc[matches["mandante_placar"].lt(matches["visitante_placar"]), "resultado_mandante"] = "Derrota"

    matches["perfil_tatico_mandante"] = matches["formacao_mandante"].map(classify_formation)
    matches["perfil_tatico_visitante"] = matches["formacao_visitante"].map(classify_formation)

    numeric_to_scale = ["cartoes_total", "cartoes_mandante", "cartoes_visitante", "total_gols_partida"]
    for column in numeric_to_scale:
        matches[f"{column}_minmax"] = minmax_scale(matches[column])
        matches[f"{column}_zscore"] = zscore_scale(matches[column])

    matches["volume_cartoes"] = pd.qcut(
        matches["cartoes_total"],
        q=4,
        labels=["Baixo", "Médio-baixo", "Médio-alto", "Alto"],
        duplicates="drop",
    )

    return matches


def build_aggregations(cards: pd.DataFrame, matches: pd.DataFrame) -> dict[str, pd.DataFrame]:
    aggregations = {}

    aggregations["agg_visao_geral"] = pd.DataFrame(
        [
            {
                "partidas": matches["partida_id"].nunique(),
                "cartoes": len(cards),
                "cartoes_amarelos": cards["cartao_amarelo"].sum(),
                "cartoes_vermelhos": cards["cartao_vermelho"].sum(),
                "media_cartoes_por_partida": matches["cartoes_total"].mean(),
                "pct_cartoes_visitante": cards["eh_visitante"].mean() * 100,
                "pct_jogos_interestaduais": matches["jogo_interestadual"].mean() * 100,
            }
        ]
    )

    monthly = (
        matches.groupby(["ano", "mes"], as_index=False)
        .agg(
            partidas=("partida_id", "nunique"),
            cartoes_total=("cartoes_total", "sum"),
            media_cartoes_por_partida=("cartoes_total", "mean"),
            vermelhos_mandante=("mandante_vermelho", "sum"),
            vermelhos_visitante=("visitante_vermelho", "sum"),
        )
        .sort_values(["ano", "mes"])
    )
    monthly["vermelhos"] = monthly["vermelhos_mandante"] + monthly["vermelhos_visitante"]
    monthly["media_movel_3m"] = monthly["media_cartoes_por_partida"].rolling(3, min_periods=1).mean()
    monthly["crescimento_pct_cartoes"] = (monthly["cartoes_total"].pct_change() * 100).fillna(0)
    aggregations["agg_cartoes_por_ano_mes"] = monthly.drop(
        columns=["vermelhos_mandante", "vermelhos_visitante"]
    )

    aggregations["agg_cartoes_mandante_visitante"] = (
        cards.groupby(["local_clube", "jogo_interestadual"], as_index=False)
        .agg(
            cartoes=("partida_id", "size"),
            amarelos=("cartao_amarelo", "sum"),
            vermelhos=("cartao_vermelho", "sum"),
            peso_medio_cartao=("cartao_peso", "mean"),
            minuto_medio=("minuto_total", "mean"),
        )
    )
    aggregations["agg_cartoes_mandante_visitante"]["participacao_pct"] = (
        aggregations["agg_cartoes_mandante_visitante"]["cartoes"]
        / aggregations["agg_cartoes_mandante_visitante"]["cartoes"].sum()
        * 100
    )

    home_formations = matches[
        ["partida_id", "formacao_mandante", "perfil_tatico_mandante"]
    ].rename(
        columns={
            "formacao_mandante": "formacao_clube",
            "perfil_tatico_mandante": "perfil_tatico_clube",
        }
    )
    home_formations["local_clube"] = "Mandante"
    away_formations = matches[
        ["partida_id", "formacao_visitante", "perfil_tatico_visitante"]
    ].rename(
        columns={
            "formacao_visitante": "formacao_clube",
            "perfil_tatico_visitante": "perfil_tatico_clube",
        }
    )
    away_formations["local_clube"] = "Visitante"
    formation_exposure = pd.concat([home_formations, away_formations], ignore_index=True)

    formation_matches = (
        formation_exposure.groupby(
            ["local_clube", "formacao_clube", "perfil_tatico_clube"], as_index=False
        )
        .agg(partidas_time=("partida_id", "nunique"))
    )
    formation_cards = (
        cards.groupby(["local_clube", "formacao_clube", "perfil_tatico_clube"], as_index=False)
        .agg(
            cartoes=("partida_id", "size"),
            vermelhos=("cartao_vermelho", "sum"),
            peso_medio_cartao=("cartao_peso", "mean"),
            minuto_medio=("minuto_total", "mean"),
        )
    )
    aggregations["agg_cartoes_por_formacao"] = (
        formation_matches.merge(
            formation_cards,
            on=["local_clube", "formacao_clube", "perfil_tatico_clube"],
            how="left",
        )
        .fillna({"cartoes": 0, "vermelhos": 0, "peso_medio_cartao": 0, "minuto_medio": 0})
        .assign(
            cartoes_por_partida_time=lambda data: data["cartoes"] / data["partidas_time"],
            vermelhos_por_partida_time=lambda data: data["vermelhos"] / data["partidas_time"],
        )
        .sort_values("cartoes_por_partida_time", ascending=False)
    )

    aggregations["agg_cartoes_por_posicao"] = (
        cards.groupby(["posicao", "cartao"], as_index=False)
        .agg(
            cartoes=("partida_id", "size"),
            minuto_medio=("minuto_total", "mean"),
            pct_reta_final=("cartao_reta_final", "mean"),
        )
    )
    aggregations["agg_cartoes_por_posicao"]["pct_reta_final"] *= 100

    aggregations["agg_cartoes_por_faixa_minuto"] = (
        cards.groupby(["faixa_minuto", "cartao"], observed=False, as_index=False)
        .agg(cartoes=("partida_id", "size"))
    )
    aggregations["agg_cartoes_por_faixa_minuto"]["participacao_pct"] = (
        aggregations["agg_cartoes_por_faixa_minuto"]["cartoes"]
        / aggregations["agg_cartoes_por_faixa_minuto"]["cartoes"].sum()
        * 100
    )

    aggregations["agg_eficiencia_disciplina"] = (
        matches.groupby(["resultado_mandante", "tipo_placar"], observed=False, as_index=False)
        .agg(
            partidas=("partida_id", "nunique"),
            media_cartoes_mandante=("cartoes_mandante", "mean"),
            media_cartoes_visitante=("cartoes_visitante", "mean"),
            media_cartoes_total=("cartoes_total", "mean"),
            pct_visitante_mais_punido=("visitante_mais_punido", "mean"),
        )
    )
    aggregations["agg_eficiencia_disciplina"]["pct_visitante_mais_punido"] *= 100

    return aggregations


def save_outputs(cards: pd.DataFrame, matches: pd.DataFrame, aggregations: dict[str, pd.DataFrame]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cards.to_csv(CARD_OUTPUT, index=False)
    matches.to_csv(MATCH_OUTPUT, index=False)
    for name, dataframe in aggregations.items():
        dataframe.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)


def main() -> None:
    clean_df = load_clean_dataset()
    cards = add_card_level_features(clean_df)
    matches = build_match_dataset(cards)
    aggregations = build_aggregations(cards, matches)
    save_outputs(cards, matches, aggregations)

    print(f"Base limpa de entrada: {clean_df.shape[0]} linhas x {clean_df.shape[1]} colunas")
    print(f"Base transformada de cartoes: {cards.shape[0]} linhas x {cards.shape[1]} colunas")
    print(f"Base transformada de partidas: {matches.shape[0]} linhas x {matches.shape[1]} colunas")
    print(f"Arquivos gerados em: {OUTPUT_DIR}")
    print("\nAgregacoes geradas:")
    for name, dataframe in aggregations.items():
        print(f"- {name}: {dataframe.shape[0]} linhas x {dataframe.shape[1]} colunas")


if __name__ == "__main__":
    main()
