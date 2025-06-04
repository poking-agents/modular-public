for i in {1..10}
do
  sleep 10
  viv run fix_repo_issue/eval-analysis-public_8@fix_viv_issue     --max_tokens 8000000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --repo modular-public --branch 2024-12-11 --commit 6cf3a080cfe51ce2bbc46a57805a064ee95ef17d --k8s=True --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sv2g_fixed_rating_c3.5sv2d_always_savea --batch-name fri_2501v2 --batch_concurrency_limit 30
done

for i in {1..5}
do
  sleep 10
  viv run fix_repo_issue/eval-analysis-public_9@fix_viv_issue     --max_tokens 8000000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --repo modular-public --branch 2024-12-11 --commit 6cf3a080cfe51ce2bbc46a57805a064ee95ef17d --k8s=True --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sv2g_fixed_rating_c3.5sv2d_always_savea --batch-name fri_2501v2 --batch_concurrency_limit 30
done

for i in {1..5}
do
  sleep 10
  viv run fix_repo_issue/eval-analysis-public_10@fix_viv_issue     --max_tokens 8000000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --repo modular-public --branch 2024-12-11 --commit 6cf3a080cfe51ce2bbc46a57805a064ee95ef17d --k8s=True --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sv2g_fixed_rating_c3.5sv2d_always_savea --batch-name fri_2501v2 --batch_concurrency_limit 30
done 

for i in {1..5}
do
  sleep 10
  viv run fix_repo_issue/eval-analysis-public_11@fix_viv_issue     --max_tokens 8000000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --repo modular-public --branch 2024-12-11 --commit 6cf3a080cfe51ce2bbc46a57805a064ee95ef17d --k8s=True --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sv2g_fixed_rating_c3.5sv2d_always_savea --batch-name fri_2501v2 --batch_concurrency_limit 30
done


 viv run lie_detector/default     --max_tokens 8000000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --repo modular-public --branch 2024-12-11 --commit 6cf3a080cfe51ce2bbc46a57805a064ee95ef17d --k8s=False --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sv2g_fixed_rating_c3.5sv2d_always_savea --batch-name throwaway --batch_concurrency_limit 30



  viv run sadservers/manhattan  --max_tokens 50000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sgda 


  viv run linguistics_olympiad/2018_team_can_verify  --max_tokens 50000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sgda  

  viv run linguistics_olympiad/2018_team_can_verify  --max_tokens 300000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sgda 

  viv run linguistics_olympiad/2018_team_can_verify  --max_tokens 2000000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sgda 
for i in {1..3}
do
  viv run implement_ace_oauth/given_models  --max_tokens 200000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sgda 
  sleep 10
done