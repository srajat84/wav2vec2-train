
			#checkpoint.save_dir=${checkpoint_path} 
			python generate_lm.py --input_txt ../lm/train.txt --output_dir ../lm/ \
				  --top_k 500000 --kenlm_bins /home/harveen.chadha/_/wav2vec2/kenlm/build/bin \
				    --arpa_order 5 --max_arpa_memory "85%" --arpa_prune "0|0|1" \
				      --binary_a_bits 255 --binary_q_bits 8 --binary_type trie

