import re
from pathlib import Path

import pandas as pd


RAW_PATH = Path("dados_brasileirao/base_integrada_projeto.csv")
CLEAN_PATH = Path("dados_brasileirao/base_integrada_projeto_limpa.csv")


def to_snake_case(column_name: str) -> str:
    column_name = column_name.strip()
    column_name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", column_name)
    column_name = re.sub(r"[^0-9a-zA-Z]+", "_", column_name)
    return column_name.strip("_").lower()


def clean_text(value):
    if pd.isna(value):
        return value

    value = str(value).replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_team_name(value):
    if pd.isna(value):
        return value

    mapping = {
        "sao paulo": "Sao Paulo",
        "são paulo": "Sao Paulo",
        "atletico mg": "Atletico-MG",
        "atlético mg": "Atletico-MG",
        "athletico pr": "Athletico-PR",
        "botafogo rj": "Botafogo-RJ",
        "america mg": "America-MG",
        "américa mg": "America-MG",
        "gremio": "Gremio",
        "grêmio": "Gremio",
        "goias": "Goias",
        "goiás": "Goias",
        "ceara": "Ceara",
        "ceará": "Ceara",
        "criciuma": "Criciuma",
        "criciúma": "Criciuma",
        "vitoria": "Vitoria",
        "vitória": "Vitoria",
        "avai": "Avai",
        "avaí": "Avai",
    }

    key = clean_text(value).lower().replace("-", " ")
    return mapping.get(key, clean_text(value).title())


def parse_money(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(r"R\$", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_match_minute(value):
    if pd.isna(value):
        return pd.NA

    match = re.fullmatch(r"(\d{1,3})(?:\+(\d{1,2}))?", str(value).strip())
    if not match:
        return pd.NA

    base_minute = int(match.group(1))
    extra_time = int(match.group(2) or 0)
    return base_minute + extra_time


def print_missing_report(df: pd.DataFrame, title: str) -> None:
    missing = (
        pd.DataFrame(
            {
                "ausentes": df.isna().sum(),
                "percentual": (df.isna().mean() * 100).round(2),
            }
        )
        .query("ausentes > 0")
        .sort_values("ausentes", ascending=False)
    )

    print(f"\n{title}")
    print(missing.to_string() if not missing.empty else "Nenhum valor ausente.")


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    summary = {
        "linhas_iniciais": len(df),
        "duplicatas_removidas": 0,
        "colunas_removidas": [],
        "datas_invalidas": 0,
        "camisas_invalidas": 0,
        "minutos_invalidos": 0,
        "linhas_com_ids_inconsistentes": 0,
        "linhas_com_placar_inconsistente": 0,
    }

    print(f"Dimensoes iniciais: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print("\nTipos iniciais")
    print(df.dtypes.to_string())
    print_missing_report(df, "Valores ausentes antes da limpeza")

    # padronizacao dos nomes de colunas para facilitar analise e futuros merges
    df = df.rename(columns={column: to_snake_case(column) for column in df.columns})
    df = df.rename(
        columns={
            "id": "partida_id",
            "rodata_x": "rodada_partida",
            "partida_id": "partida_id_cartao",
            "rodata_y": "rodada_cartao",
        }
    )

    # remocao de duplicatas completas
    duplicated_rows = df.duplicated().sum()
    print(f"\nDuplicatas completas identificadas: {duplicated_rows}")
    df = df.drop_duplicates().copy()
    summary["duplicatas_removidas"] = int(duplicated_rows)

    # padronizacao de textos e categorias
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        df[column] = df[column].map(clean_text)

    team_columns = ["mandante", "visitante", "vencedor", "clube"]
    for column in team_columns:
        df[column] = df[column].map(
            lambda value: "-" if value == "-" else normalize_team_name(value)
        )

    state_columns = ["mandante_estado", "visitante_estado"]
    for column in state_columns:
        df[column] = df[column].astype("string").str.upper().map(clean_text)

    categorical_title_columns = [
        "cartao",
        "posicao",
        "formacao_mandante",
        "formacao_visitante",
        "arena",
    ]
    for column in categorical_title_columns:
        df[column] = df[column].map(
            lambda value: value if pd.isna(value) else clean_text(value).title()
        )

    position_mapping = {"Zagueira": "Zagueiro"}
    df["posicao"] = df["posicao"].replace(position_mapping)

    # padronizacao de datas para permitir filtros temporais e series historicas
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y", errors="coerce")
    summary["datas_invalidas"] = int(df["data"].isna().sum())

    future_dates = df["data"] > pd.Timestamp.today().normalize()
    print(f"Datas futuras identificadas: {future_dates.sum()}")
    df.loc[future_dates, "data"] = pd.NaT

    # correcao de tipos numericos e monetarios
    id_columns = ["partida_id", "partida_id_cartao"]
    integer_columns = [
        "partida_id",
        "partida_id_cartao",
        "rodada_partida",
        "rodada_cartao",
        "mandante_placar",
        "visitante_placar",
        "num_camisa",
    ]

    for column in integer_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["arrecadacao"] = parse_money(df["arrecadacao"])

    id_inconsistent = df["partida_id"].ne(df["partida_id_cartao"])
    round_inconsistent = df["rodada_partida"].ne(df["rodada_cartao"])
    summary["linhas_com_ids_inconsistentes"] = int(id_inconsistent.sum())
    print(f"Linhas com IDs divergentes entre partidas e cartoes: {id_inconsistent.sum()}")
    print(f"Linhas com rodadas divergentes entre partidas e cartoes: {round_inconsistent.sum()}")

    # colunas redundantes da integracao sao removidas apos validar equivalencia
    if not id_inconsistent.any():
        df = df.drop(columns=["partida_id_cartao"])
        summary["colunas_removidas"].append("partida_id_cartao")
    if not round_inconsistent.any():
        df = df.drop(columns=["rodada_cartao"])
        summary["colunas_removidas"].append("rodada_cartao")

    # tratamento de valores ausentes em colunas numericas
    missing_arrecadacao_pct = df["arrecadacao"].isna().mean() * 100
    print(f"Percentual ausente em arrecadacao: {missing_arrecadacao_pct:.2f}%")
    if missing_arrecadacao_pct > 80:
        df = df.drop(columns=["arrecadacao"])
        summary["colunas_removidas"].append("arrecadacao")

    shirt_missing = df["num_camisa"].isna().sum()
    print(f"Camisas ausentes preenchidas com 0: {shirt_missing}")
    df["num_camisa"] = df["num_camisa"].fillna(0)

    # tratamento de valores ausentes em colunas categoricas
    categorical_columns = df.select_dtypes(include=["object", "string"]).columns
    df[categorical_columns] = df[categorical_columns].fillna("Não informado")

    # tratamento de inconsistencias numericas conforme contexto do futebol
    invalid_scores = (df["mandante_placar"] < 0) | (df["visitante_placar"] < 0)
    print(f"Linhas com placar negativo: {invalid_scores.sum()}")
    df = df.loc[~invalid_scores].copy()

    invalid_shirts = (df["num_camisa"] < 0) | (df["num_camisa"] > 99)
    summary["camisas_invalidas"] = int(invalid_shirts.sum())
    print(f"Camisas fora do intervalo 1-99: {invalid_shirts.sum()}")
    df.loc[invalid_shirts, "num_camisa"] = 0

    df["minuto_total"] = df["minuto"].map(parse_match_minute)
    invalid_minutes = df["minuto_total"].isna() | (df["minuto_total"] < 0) | (df["minuto_total"] > 130)
    summary["minutos_invalidos"] = int(invalid_minutes.sum())
    print(f"Minutos invalidos ou fora de 0-130: {invalid_minutes.sum()}")
    df.loc[invalid_minutes, ["minuto", "minuto_total"]] = ["Não informado", pd.NA]

    valid_winner = (
        df["vencedor"].eq("-")
        | df["vencedor"].eq(df["mandante"])
        | df["vencedor"].eq(df["visitante"])
    )
    score_winner_consistent = (
        ((df["mandante_placar"] > df["visitante_placar"]) & df["vencedor"].eq(df["mandante"]))
        | ((df["mandante_placar"] < df["visitante_placar"]) & df["vencedor"].eq(df["visitante"]))
        | ((df["mandante_placar"] == df["visitante_placar"]) & df["vencedor"].eq("-"))
    )
    inconsistent_score = ~valid_winner | ~score_winner_consistent
    summary["linhas_com_placar_inconsistente"] = int(inconsistent_score.sum())
    print(f"Linhas com vencedor incoerente com placar: {inconsistent_score.sum()}")
    df = df.loc[~inconsistent_score].copy()

    for column in ["partida_id", "rodada_partida", "mandante_placar", "visitante_placar", "num_camisa", "minuto_total"]:
        df[column] = df[column].astype("Int64")

    df["ano"] = df["data"].dt.year.astype("Int64")
    df["mes"] = df["data"].dt.month.astype("Int64")

    summary["linhas_finais"] = len(df)
    return df, summary


def main() -> None:
    df = load_dataset(RAW_PATH)
    df_clean, summary = clean_dataset(df)
    df_clean.to_csv(CLEAN_PATH, index=False)

    print_missing_report(df_clean, "Valores ausentes depois da limpeza")
    print("\nTipos finais")
    print(df_clean.dtypes.to_string())
    print(f"\nDimensoes finais: {df_clean.shape[0]} linhas x {df_clean.shape[1]} colunas")
    print(f"Arquivo limpo gerado em: {CLEAN_PATH}")
    print("\nResumo da limpeza")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
