import argparse

import polars as pl
import gff3_parser



def parse_args():
    parser = argparse.ArgumentParser(prog="", usage="")

    parser.add_argument("-g", "--gff", required=True, help="GFF3 to parse transcripts from.")
    parser.add_argument("-o", "--output", required=True, help="Output file.")

    return parser.parse_args()


def parse_gff(path: str) -> pl.DataFrame:
    pandas_df = gff3_parser.parse_gff3(path, parse_attributes = True)
    return pl.from_pandas(pandas_df) \
        .rename({"Seqid": "Chrom"})


def main():
    args = parse_args()

    df_gff = parse_gff(args.gff)

    df_genes = df_gff \
        .filter((pl.col("Type") == "gene") & (pl.col("biotype") == "protein_coding")) \
        .select(["gene_id", "Name"]) \
        .rename({"Name": "gene_symbol"})

    df_transcripts = df_gff \
        .filter(pl.col("Type") == "mRNA") \
        .select(["transcript_id", "Parent"]) \
        .with_columns(pl.col("Parent").str.strip_prefix("gene:").alias("gene_id")) \
        .drop("Parent")

    df_exons = df_gff.filter(pl.col("Type") == "exon") \
        .select(["Chrom", "Start", "End", "Strand", "exon_id", "Parent"]) \
        .with_columns(
            pl.col("Parent").str.strip_prefix("transcript:").alias("transcript_id"),
            pl.col("Start").cast(pl.Int128).alias("Start"),
            pl.col("End").cast(pl.Int128).alias("End")
        ) \
        .drop("Parent")

    df_coding_exons = df_exons \
        .join(df_transcripts, how="inner", on="transcript_id") \
        .join(df_genes, how="inner", on="gene_id") \
        .sort(["Chrom", "Start", "End"])

    df = df_coding_exons \
        .group_by(["Chrom", "gene_id", "gene_symbol", "Strand"]) \
        .agg(pl.col("Start").min(), pl.col("End").max()) \
        .rename({"Chrom": "CHROM", "gene_symbol": "NAME", "Start": "TX_START", "End": "TX_END", "Strand": "STRAND"}) \
        .drop("gene_id") \
        .select(["NAME", "CHROM", "STRAND", "TX_START", "TX_END"]) \
        .sort(["CHROM", "TX_START", "TX_END"]) \
        .with_columns((pl.col("CHROM")).alias("CHROM"))

    df.write_csv(args.output, separator="\t")


if __name__ == "__main__":
    main()
