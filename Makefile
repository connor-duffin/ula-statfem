PYTHON3 = python3

NX_SMALL = 32
OUTPUT_DIR_PRIOR_SMALL = outputs/prior-mesh-$(NX_SMALL)
OUTPUT_DIR_POST_SMALL = outputs/posterior-mesh-ll-$(NX_SMALL)
DATA_FILE_SMALL = $(OUTPUT_DIR_POST_SMALL)/data.h5

NX = 128
OUTPUT_DIR_PRIOR = outputs/prior-mesh-$(NX)
OUTPUT_DIR_POST = outputs/posterior-mesh-ll-$(NX)
OUTPUT_DIR_NLL = outputs/posterior-mesh-nll-128
DATA_FILE = $(OUTPUT_DIR_POST)/data.h5



# low-dimensional prior (laptop-friendly)
$(OUTPUT_DIR_PRIOR_SMALL)/pula.h5:
	python3 scripts/run_samplers_2d.py \
		--sampler pula-lu --eta 2e-1 \
		--nx $(NX_SMALL) --n_sample 1000 --n_inner 10 --n_warmup 0 --cold_start \
		--output_file $@

$(OUTPUT_DIR_PRIOR_SMALL)/pula-smallstep.h5:
	python3 scripts/run_samplers_2d.py \
		--sampler pula-lu --eta 1e-3 \
		--nx $(NX_SMALL) --n_sample 1000 --n_inner 10 --n_warmup 0 --cold_start \
		--output_file $@

$(OUTPUT_DIR_PRIOR_SMALL)/mala.h5:
	python3 scripts/run_samplers_2d.py \
		--sampler mala --eta 1e-8 \
		--nx $(NX_SMALL) --n_sample 1000 --n_inner 10 --n_warmup 0 --cold_start \
		--output_file $@

$(OUTPUT_DIR_PRIOR_SMALL)/pmala.h5:
	python3 scripts/run_samplers_2d.py \
		--sampler pmala \
		--nx $(NX_SMALL) --n_sample 1000 --n_inner 10 --n_warmup 0 --cold_start \
		--output_file $@

$(OUTPUT_DIR_PRIOR_SMALL)/exact.h5:
	python3 scripts/run_samplers_2d.py \
		--sampler exact \
		--nx $(NX_SMALL) --n_sample 1000 --output_file $@

all_samplers_prior_small: $(OUTPUT_DIR_PRIOR_SMALL)/pula.h5 \
	$(OUTPUT_DIR_PRIOR_SMALL)/pula-smallstep.h5 \
	$(OUTPUT_DIR_PRIOR_SMALL)/mala.h5 \
	$(OUTPUT_DIR_PRIOR_SMALL)/pmala.h5 \
	$(OUTPUT_DIR_PRIOR_SMALL)/exact.h5

clean_output_dir_prior_small:
	rm -rf $(OUTPUT_DIR_PRIOR_SMALL)/*

plots_prior_paper_small: scripts/plot_prior_paper_small.py
	python3 scripts/plot_prior_paper_small.py \
		--nx $(NX_SMALL) \
		--input_dir $(OUTPUT_DIR_PRIOR_SMALL)/ \
		--output_dir figures/prior-mesh-$(NX_SMALL)/

plots_prior_small: all_samplers_prior_small scripts/plot_samplers_2d.py
	python3 scripts/plot_samplers_2d.py \
		--prior --nx $(NX_SMALL) --n_warmup 0 \
		--input_dir $(OUTPUT_DIR_PRIOR_SMALL)/ --output_dir figures/prior-mesh-$(NX_SMALL)/



# generate the data
$(DATA_FILE_SMALL):
	python3 scripts/generate_data_2d.py --output_file $@

# low-dimensional posterior
$(OUTPUT_DIR_POST_SMALL)/ula.h5: $(DATA_FILE_SMALL)
	python3 scripts/run_samplers_2d_posterior.py \
		--sampler ula --eta 1e-8 \
		--nx $(NX_SMALL) --n_sample 15000 --n_warmup 5000 --n_inner 10 \
		--output_file $@ --data_file $(DATA_FILE_SMALL)

$(OUTPUT_DIR_POST_SMALL)/pula.h5: $(DATA_FILE_SMALL)
	python3 scripts/run_samplers_2d_posterior.py \
		--sampler pula \
		--nx $(NX_SMALL) --n_sample 15000 --n_warmup 5000 --n_inner 10 \
		--output_file $@ --data_file $(DATA_FILE_SMALL)

$(OUTPUT_DIR_POST_SMALL)/mala.h5: $(DATA_FILE_SMALL)
	python3 scripts/run_samplers_2d_posterior.py \
		--sampler mala --eta 1e-9 \
		--nx $(NX_SMALL) --n_sample 15000 --n_warmup 5000 --n_inner 10 \
		--output_file $@ --data_file $(DATA_FILE_SMALL)

$(OUTPUT_DIR_POST_SMALL)/pmala.h5: $(DATA_FILE_SMALL)
	python3 scripts/run_samplers_2d_posterior.py \
		--sampler pmala \
		--nx $(NX_SMALL) --n_sample 15000 --n_warmup 5000 --n_inner 10 \
		--output_file $@ --data_file $(DATA_FILE_SMALL)

$(OUTPUT_DIR_POST_SMALL)/pcn.h5: $(DATA_FILE_SMALL)
	python3 scripts/run_samplers_2d_posterior.py \
		--sampler pcn  --eta 1e-2 \
		--nx $(NX_SMALL) --n_sample 15000 --n_warmup 5000 --n_inner 10 \
		--output_file $@ --data_file $(DATA_FILE_SMALL)

$(OUTPUT_DIR_POST_SMALL)/exact.h5: $(DATA_FILE_SMALL)
	python3 scripts/run_samplers_2d_posterior.py \
		--sampler exact \
		--nx $(NX_SMALL) --n_sample 10000 \
		--output_file $@ --data_file $(DATA_FILE_SMALL)

all_samplers_post_small: $(OUTPUT_DIR_POST_SMALL)/ula.h5 \
	$(OUTPUT_DIR_POST_SMALL)/pula.h5 \
	$(OUTPUT_DIR_POST_SMALL)/mala.h5 \
	$(OUTPUT_DIR_POST_SMALL)/pmala.h5 \
	$(OUTPUT_DIR_POST_SMALL)/exact.h5 \
	$(OUTPUT_DIR_POST_SMALL)/pcn.h5

clean_output_dir_post_small:
	rm -rf $(OUTPUT_DIR_POST_SMALL)/*

plots_post_paper_small: scripts/plot_post_paper.py
	python3 $< \
		--nx 32 --n_warmup 5000 \
		--input_dir outputs/posterior-mesh-ll-32/ \
		--output_dir figures/posterior-mesh-32/

# n_warmup = 0 as we don't save warmup iterations anymore
# this is a legacy option that is here for posterity
plots_post_small: all_samplers_post_small scripts/plot_samplers_2d.py
	python3 scripts/plot_samplers_2d.py \
		--nx $(NX_SMALL) --n_warmup 500 \
		--input_dir outputs/posterior-mesh-ll-32/ --output_dir figures/posterior-mesh-32/

# nonlinear likelihood example
# ----------------------------
N_WARMUP = 20000
N_SAMPLE = 40000

# nonlinear observation processs
$(OUTPUT_DIR_NLL)/data.h5: scripts/generate_data_2d.py
	python3 $< --nonlinear_observation --output_file $@

# exact samples: 200_000 samples from pMALA
$(OUTPUT_DIR_NLL)/exact.h5: scripts/run_samplers_2d_posterior.py \
	$(OUTPUT_DIR_NLL)/data.h5
	python3 $< --sampler pmala --nonlinear_observation \
		--nx $(NX) --n_sample 220000 --n_warmup $(N_WARMUP) --n_inner 10 \
		--data_file $(OUTPUT_DIR_NLL)/data.h5 --output_file $@

$(OUTPUT_DIR_NLL)/ula.h5: scripts/run_samplers_2d_posterior.py \
	$(OUTPUT_DIR_NLL)/data.h5
	python3 $< --sampler ula --nonlinear_observation \
		--eta 1e-9 --nx $(NX) --n_sample $(N_SAMPLE) --n_warmup $(N_WARMUP) --n_inner 50 \
		--data_file $(OUTPUT_DIR_NLL)/data.h5 --output_file $@

$(OUTPUT_DIR_NLL)/pula.h5: scripts/run_samplers_2d_posterior.py \
	$(OUTPUT_DIR_NLL)/data.h5
	python3 $< --sampler pula --nonlinear_observation \
		--nx $(NX) --n_sample $(N_SAMPLE) --n_warmup $(N_WARMUP) --n_inner 10 \
		--data_file $(OUTPUT_DIR_NLL)/data.h5 --output_file $@

$(OUTPUT_DIR_NLL)/mala.h5: scripts/run_samplers_2d_posterior.py \
	$(OUTPUT_DIR_NLL)/data.h5
	python3 $< --sampler mala --nonlinear_observation \
		--eta 1e-10 --nx $(NX) --n_sample $(N_SAMPLE) --n_warmup $(N_WARMUP) --n_inner 10 \
		--data_file $(OUTPUT_DIR_NLL)/data.h5 --output_file $@

$(OUTPUT_DIR_NLL)/pmala.h5: scripts/run_samplers_2d_posterior.py \
	$(OUTPUT_DIR_NLL)/data.h5
	python3 $< --sampler pmala --nonlinear_observation \
		--nx $(NX) --n_sample $(N_SAMPLE) --n_warmup $(N_WARMUP) --n_inner 10 \
		--data_file $(OUTPUT_DIR_NLL)/data.h5 --output_file $@

$(OUTPUT_DIR_NLL)/pcn.h5: scripts/run_samplers_2d_posterior.py \
	$(OUTPUT_DIR_NLL)/data.h5
	python3 $< --sampler pcn --nonlinear_observation \
		--eta 1e-3 --nx $(NX) --n_sample $(N_SAMPLE) --n_warmup $(N_WARMUP) --n_inner 10 \
		--data_file $(OUTPUT_DIR_NLL)/data.h5 --output_file $@

all_samplers_nll_post: $(OUTPUT_DIR_NLL)/ula.h5 \
	$(OUTPUT_DIR_NLL)/pula.h5 \
	$(OUTPUT_DIR_NLL)/mala.h5 \
	$(OUTPUT_DIR_NLL)/pmala.h5

clean_all_samplers_nll_post:
	rm $(OUTPUT_DIR_NLL)/ula.h5 \
			$(OUTPUT_DIR_NLL)/pula.h5 \
			$(OUTPUT_DIR_NLL)/mala.h5 \
			$(OUTPUT_DIR_NLL)/pmala.h5

plots_post_paper_nll: scripts/plot_samplers_2d.py
	python3 scripts/plot_post_paper.py \
		--nx $(NX) --n_warmup 20000 \
		--file_ids pula pmala ula mala pcn \
		--input_dir $(OUTPUT_DIR_NLL)/ \
		--output_dir figures/posterior-mesh-nll-128/


# high-dimensional prior
$(OUTPUT_DIR_PRIOR)/pula.h5:
	python3 scripts/run_samplers_2d.py \
		--sampler pula-lu \
		--nx $(NX) --n_sample 10000 --n_inner 10 --output_file $@

$(OUTPUT_DIR_PRIOR)/mala.h5:
	python3 scripts/run_samplers_2d.py \
		--sampler mala --eta 1e-8 \
		--nx $(NX) --n_sample 15000 --n_inner 10 --n_warmup 5000 --output_file $@

$(OUTPUT_DIR_PRIOR)/pmala.h5:
	python3 scripts/run_samplers_2d.py \
		--sampler pmala \
		--nx $(NX) --n_sample 15000 --n_inner 10 --n_warmup 5000 --output_file $@

$(OUTPUT_DIR_PRIOR)/exact.h5:
	python3 scripts/run_samplers_2d.py \
		--sampler exact \
		--nx $(NX) --n_sample 10000 --output_file $@

all_samplers_prior: $(OUTPUT_DIR_PRIOR)/exact.h5 \
	$(OUTPUT_DIR_PRIOR)/pula.h5 \
	$(OUTPUT_DIR_PRIOR)/mala.h5 \
	$(OUTPUT_DIR_PRIOR)/pmala.h5

clean_output_dir_prior:
	rm -rf $(OUTPUT_DIR_PRIOR)/*

# n_warmup = 0 as we only store post-warmup samples (legacy option)
plots_prior: scripts/plot_samplers_2d.py
	python3 scripts/plot_samplers_2d.py \
		--prior --nx $(NX) --n_warmup 0 \
		--input_dir $(OUTPUT_DIR_PRIOR)/ --output_dir figures/prior-mesh-${NX}/

plots_prior_paper: scripts/plot_prior_paper.py
	python3 $< \
		--nx 128 --n_warmup 0 \
		--input_dir $(OUTPUT_DIR_PRIOR)/ \
		--output_dir figures/prior-mesh-${NX}/


# high-dimensional posterior
$(DATA_FILE):
	python3 scripts/generate_data_2d.py --output_file $@

$(OUTPUT_DIR_POST)/ula.h5: $(DATA_FILE)
	python3 scripts/run_samplers_2d_posterior.py \
		--sampler ula --eta 1e-9 \
		--nx $(NX) --n_sample 20000 --n_warmup 10000 --n_inner 10 \
		--output_file $@ --data_file $(DATA_FILE)

$(OUTPUT_DIR_POST)/mala.h5: $(DATA_FILE)
	python3 scripts/run_samplers_2d_posterior.py \
		--sampler mala --eta 1e-9 \
		--nx $(NX) --n_sample 20000 --n_warmup 10000 --n_inner 10 \
		--output_file $@ --data_file $(DATA_FILE)

$(OUTPUT_DIR_POST)/pula.h5: $(DATA_FILE)
	python3 scripts/run_samplers_2d_posterior.py \
		--sampler pula \
		--nx $(NX) --n_sample 20000 --n_warmup 10000 --n_inner 10 \
		--output_file $@ --data_file $(DATA_FILE)

$(OUTPUT_DIR_POST)/pmala.h5: $(DATA_FILE)
	python3 scripts/run_samplers_2d_posterior.py \
		--sampler pmala \
		--nx $(NX) --n_sample 20000 --n_warmup 10000 --n_inner 10 \
		--output_file $@ --data_file $(DATA_FILE)

$(OUTPUT_DIR_POST)/pcn.h5: $(DATA_FILE)
	python3 scripts/run_samplers_2d_posterior.py \
		--sampler pcn  --eta 1e-2 \
		--nx $(NX) --n_sample 20000 --n_warmup 10000 --n_inner 10 \
		--output_file $@ --data_file $(DATA_FILE)

$(OUTPUT_DIR_POST)/exact.h5: $(DATA_FILE)
	python3 scripts/run_samplers_2d_posterior.py \
		--sampler exact \
		--nx $(NX) --n_sample 10000 \
		--output_file $@ --data_file $(DATA_FILE)

all_samplers_post: $(OUTPUT_DIR_POST)/ula.h5 \
	$(OUTPUT_DIR_POST)/pula.h5 \
	$(OUTPUT_DIR_POST)/mala.h5 \
	$(OUTPUT_DIR_POST)/pmala.h5 \
	$(OUTPUT_DIR_POST)/exact.h5 \
	$(OUTPUT_DIR_POST)/pcn.h5

plots_post: all_samplers_post scripts/plot_samplers_2d.py
	python3 scripts/plot_samplers_2d.py \
		--nx $(NX) --n_warmup 0 \
		--input_dir outputs/posterior-mesh-ll-128/ \
		--output_dir figures/posterior-mesh-128/

plots_post_paper: scripts/plot_post_paper.py
	python3 $< \
		--nx 128 --n_warmup 0 \
		--file_ids pula pmala ula mala \
		--input_dir outputs/posterior-mesh-ll-128/ \
		--output_dir figures/posterior-mesh-128/

clean_output_dir_post:
	rm -rf $(OUTPUT_DIR_POST)/*
