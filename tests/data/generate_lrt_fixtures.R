#!/usr/bin/env Rscript

# Generate DESeq2 1.34.0 reference fixtures for PyDESeq2's classical LRT.

suppressPackageStartupMessages(library(DESeq2))

required_deseq2_version <- package_version("1.34.0")
installed_deseq2_version <- packageVersion("DESeq2")
if (installed_deseq2_version != required_deseq2_version) {
    stop(
        sprintf(
            "Expected DESeq2 %s, found %s.",
            required_deseq2_version,
            installed_deseq2_version
        )
    )
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) {
    stop("Could not determine the path to generate_lrt_fixtures.R.")
}
script_path <- normalizePath(sub("^--file=", "", script_arg[[1L]]), mustWork = TRUE)
repo_root <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = TRUE)

counts_path <- file.path(repo_root, "datasets", "synthetic", "test_counts.csv")
metadata_path <- file.path(repo_root, "datasets", "synthetic", "test_metadata.csv")
output_dir <- file.path(repo_root, "tests", "data", "lrt")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# The repository CSV stores genes in rows and samples in columns, matching DESeq2.
count_data <- as.matrix(
    read.csv(counts_path, row.names = 1L, check.names = FALSE)
)
storage.mode(count_data) <- "integer"
col_data <- read.csv(metadata_path, row.names = 1L, check.names = FALSE)

if (!setequal(colnames(count_data), rownames(col_data))) {
    stop("Sample names in the synthetic counts and metadata do not match.")
}
col_data <- col_data[colnames(count_data), , drop = FALSE]
stopifnot(identical(colnames(count_data), rownames(col_data)))

col_data$condition <- factor(col_data$condition, levels = c("A", "B"))
col_data$group <- factor(col_data$group, levels = c("X", "Y"))

write_results <- function(results_object, filename) {
    output <- as.data.frame(results_object)
    output <- output[rownames(count_data), , drop = FALSE]
    write.csv(
        output,
        file.path(output_dir, filename),
        quote = FALSE,
        na = "NA"
    )
}

run_lrt <- function(
    counts,
    metadata,
    full,
    reduced,
    contrast,
    filename,
    seed,
    ...
) {
    set.seed(seed)
    dds <- DESeqDataSetFromMatrix(
        countData = counts,
        colData = metadata,
        design = full
    )
    dds <- DESeq(
        dds,
        test = "LRT",
        reduced = reduced,
        fitType = "parametric",
        sfType = "ratio",
        betaPrior = FALSE,
        quiet = TRUE,
        ...
    )
    write_results(
        results(
            dds,
            contrast = contrast,
            alpha = 0.05,
            independentFiltering = TRUE,
            cooksCutoff = TRUE
        ),
        filename
    )
    invisible(dds)
}

run_lrt(
    count_data,
    col_data,
    full = ~condition,
    reduced = ~1,
    contrast = c("condition", "B", "A"),
    filename = "r_lrt_single_factor.csv",
    seed = 1001L
)

run_lrt(
    count_data,
    col_data,
    full = ~group + condition,
    reduced = ~group,
    contrast = c("condition", "B", "A"),
    filename = "r_lrt_multi_factor.csv",
    seed = 1002L
)

multilevel_data <- col_data
multilevel_data$condition3 <- factor(
    rep(c("A", "B", "C"), length.out = nrow(multilevel_data)),
    levels = c("A", "B", "C")
)
run_lrt(
    count_data,
    multilevel_data,
    full = ~condition3,
    reduced = ~1,
    contrast = c("condition3", "B", "A"),
    filename = "r_lrt_multilevel.csv",
    seed = 1003L
)

outlier_counts <- count_data
outlier_counts["gene1", "sample1"] <- 200L
outlier_dds <- run_lrt(
    outlier_counts,
    col_data,
    full = ~condition,
    reduced = ~1,
    contrast = c("condition", "B", "A"),
    filename = "r_lrt_outlier.csv",
    seed = 1004L,
    minReplicatesForReplace = 7L
)

if (!"replaceCounts" %in% assayNames(outlier_dds)) {
    stop("The outlier fixture did not produce a replaceCounts assay.")
}
replaced_genes <- mcols(outlier_dds)$replace
gene1_index <- match("gene1", rownames(outlier_dds))
if (is.null(replaced_genes) || !isTRUE(replaced_genes[[gene1_index]])) {
    stop("DESeq2 did not flag gene1 for outlier replacement as expected.")
}
write.csv(
    assay(outlier_dds, "replaceCounts"),
    file.path(output_dir, "r_lrt_outlier_replace_counts.csv"),
    quote = FALSE
)

capture.output(
    sessionInfo(),
    file = file.path(output_dir, "lrt_session_info.txt")
)

message("Wrote LRT fixtures to ", output_dir)
