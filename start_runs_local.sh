n=15
for i in $(seq 1 $n)
do
  sleep 2
  echo "Running more_bias_scrambled_folders@lrt_experiments"
  viv run local_research_tex/more_bias_scrambled_folders@lrt_experiments  --max_tokens 5000000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --repo modular-public --branch 2024-12-11 --commit 6cf3a080cfe51ce2bbc46a57805a064ee95ef17d --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sv2g_fixed_rating_c3.5sv2d_always_savea  --k8s=True --batch-name lrt_experiments_2501v1 --batch_concurrency_limit 30
done
exit 1

for i in $(seq 1 $n)
do
  sleep 2
  echo "Running more_bias_more_arxiv@lrt_experiments"
  viv run local_research_tex/more_bias_more_arxiv_50@lrt_experiments    --max_tokens 5000000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --repo modular-public --branch 2024-12-11 --commit 6cf3a080cfe51ce2bbc46a57805a064ee95ef17d --agent_settings_pack  t_context_and_usage_awarep_claude_legacy_1xc3.5sv2g_fixed_rating_c3.5sv2d_always_savea --k8s=True  --batch-name lrt_experiments_2501v1 --batch_concurrency_limit 30
done

for i in $(seq 1 $n)
do
  sleep 2
  echo "Running more_bias_more_arxiv@lrt_experiments"
  viv run local_research_tex/more_bias_scrambled_files@lrt_experiments    --max_tokens 5000000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --repo modular-public --branch 2024-12-11 --commit 6cf3a080cfe51ce2bbc46a57805a064ee95ef17d --agent_settings_pack  t_context_and_usage_awarep_claude_legacy_1xc3.5sv2g_fixed_rating_c3.5sv2d_always_savea --k8s=True  --batch-name lrt_experiments_2501v1 --batch_concurrency_limit 30
done
exit 1
for i in $(seq 1 $n)
do
  #sleep 3
  echo "Running more_bias_more_arxiv@lrt_experiments"
  #viv run local_research_tex/more_bias_more_arxiv@lrt_experiments    --max_tokens 5000000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --repo modular-public --branch 2024-12-11 --commit 6cf3a080cfe51ce2bbc46a57805a064ee95ef17d --agent_settings_pack  t_context_and_usage_awarep_claude_legacy_1xc3.5sv2g_fixed_rating_c3.5sv2d_always_savea --k8s=True  --batch-name lrt_experiments_2501v1 --batch_concurrency_limit 30
done
for i in $(seq 1 $n)
do
  sleep 2
  echo "Running more_bias_random_files@lrt_experiments"
  viv run local_research_tex/more_bias_random_files@lrt_experiments    --max_tokens 5000000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --repo modular-public --branch 2024-12-11 --commit 6cf3a080cfe51ce2bbc46a57805a064ee95ef17d --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sv2g_fixed_rating_c3.5sv2d_always_savea --k8s=True  --batch-name lrt_experiments_2501v1 --batch_concurrency_limit 30
done
for i in $(seq 1 $n)
do
  sleep 3
  echo "Running more_bias_scrambled_folders@lrt_experiments"
  viv run local_research_tex/more_bias_scrambled_folders@lrt_experiments  --max_tokens 5000000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --repo modular-public --branch 2024-12-11 --commit 6cf3a080cfe51ce2bbc46a57805a064ee95ef17d --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sv2g_fixed_rating_c3.5sv2d_always_savea  --k8s=True --batch-name lrt_experiments_2501v1 --batch_concurrency_limit 30
done
for i in $(seq 1 $n)
do
  sleep 2
  echo "Running more_bias@lrt_experiments"
  viv run local_research_tex/more_bias@lrt_experiments  --max_tokens 5000000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --repo modular-public --branch 2024-12-11 --commit 6cf3a080cfe51ce2bbc46a57805a064ee95ef17d --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sv2g_fixed_rating_c3.5sv2d_always_savea  --k8s=True --batch-name lrt_experiments_2501v1 --batch_concurrency_limit 30
done
