
			#checkpoint.save_dir=${checkpoint_path} 
			python generate_lm.py --input_txt ../../data/lm/train.txt --output_dir ../../data/lm \
				  --top_k 500000 --kenlm_bins /opt/kenlm/build/bin \
				    --arpa_order 3 --max_arpa_memory "85%"  --arpa_prune "0|0|1" --discount_fallback  \
				      --binary_a_bits 255 --binary_q_bits 8 --binary_type trie

