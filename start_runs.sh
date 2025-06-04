for i in {1..5}
do
  sleep 10
  viv run fix_repo_issue/eval-analysis-public_8    --max_tokens 8000000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --repo modular-public --branch 2024-12-11 --commit 6cf3a080cfe51ce2bbc46a57805a064ee95ef17d --k8s=True --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sv2g_fixed_rating_c3.5sv2d_always_savea  --task_family_path ../mp4-tasks/fix_repo_issue --env_file_path ../mp4-tasks/secrets.env --batch-name fri_2501v2 --batch_concurrency_limit 30
done

for i in {1..5}
do
  sleep 10
  viv run fix_repo_issue/eval-analysis-public_9    --max_tokens 8000000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --repo modular-public --branch 2024-12-11 --commit 6cf3a080cfe51ce2bbc46a57805a064ee95ef17d --k8s=True --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sv2g_fixed_rating_c3.5sv2d_always_savea  --task_family_path ../mp4-tasks/fix_repo_issue --env_file_path ../mp4-tasks/secrets.env --batch-name fri_2501v2 --batch_concurrency_limit 30
done

for i in {1..5}
do
  sleep 10
  viv run fix_repo_issue/eval-analysis-public_10    --max_tokens 8000000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --repo modular-public --branch 2024-12-11 --commit 6cf3a080cfe51ce2bbc46a57805a064ee95ef17d --k8s=True --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sv2g_fixed_rating_c3.5sv2d_always_savea  --task_family_path ../mp4-tasks/fix_repo_issue --env_file_path ../mp4-tasks/secrets.env --batch-name fri_2501v2 --batch_concurrency_limit 30
done 

for i in {1..5}
do
  sleep 10
  viv run fix_repo_issue/eval-analysis-public_11    --max_tokens 8000000 --max_actions 2500 --max_total_seconds 36000 --max_cost 100 --repo modular-public --branch 2024-12-11 --commit 6cf3a080cfe51ce2bbc46a57805a064ee95ef17d --k8s=True --agent_settings_pack t_context_and_usage_awarep_claude_legacy_1xc3.5sv2g_fixed_rating_c3.5sv2d_always_savea  --task_family_path ../mp4-tasks/fix_repo_issue --env_file_path ../mp4-tasks/secrets.env --batch-name fri_2501v2 --batch_concurrency_limit 30
done
