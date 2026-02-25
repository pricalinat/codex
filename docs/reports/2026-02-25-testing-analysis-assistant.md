# 测试分析助手（MVP）交付文档

## 1. 你现在拿到的能力
已实现一个可运行的测试分析助手，包含：
- `unittest -v` 输出解析（支持单行和两行 verbose 结果）
- 失败项结构化提取（测试名、模块、错误类型、摘要）
- 风险分级（P0/P1/P2）
- 模块维度统计
- 风险分数（0-100）
- 建议动作生成（执行命令级）
- Markdown + JSON 报告输出
- 与上一份 JSON 报告的趋势对比

## 2. 代码位置
- 核心分析器：`src/multi_agent/test_analysis.py`
- 命令入口：`src/multi_agent/test_analysis_cli.py`
- 分析器测试：`tests/test_test_analysis.py`

## 3. 运行方式
在仓库根目录执行：

```bash
python3 -m src.multi_agent.test_analysis_cli \
  --workdir /Users/rrp/Documents/codex \
  --report-dir /Users/rrp/Documents/codex/docs/reports \
  --basename latest-test-analysis
```

## 4. 输出产物
默认生成两份文件：
- `docs/reports/latest-test-analysis.md`（人读）
- `docs/reports/latest-test-analysis.json`（机器处理）

当前最新报告：
- `docs/reports/latest-test-analysis.md`

## 5. 当前项目实测结果
本次运行解析到：
- 总测试：36
- 通过：36
- 失败：0
- 错误：0
- 风险分数：0
- 趋势：baseline report created

## 6. 你可以立即做的集成
- 本地开发：每次改动后执行 CLI，自动生成报告。
- CI：在测试步骤后追加一次 CLI 运行，并上传 `md/json` 作为构建产物。
- PR 审查：把 `latest-test-analysis.md` 作为质量结论附件。

## 7. 后续增强建议（下一迭代）
- 接入 coverage 数据（行/分支覆盖）并加入风险评分。
- 引入 flaky 检测（跨历史报告统计“偶发失败”测试）。
- 增加“失败聚类”视图（同根因归并）。
- 增加失败定位链接（文件 + 行号）。

## 8. 验收状态
- `python3 -m unittest -v` 全量通过（36/36）
- 分析器单测通过（`tests/test_test_analysis.py`）
- 报告可稳定生成并含趋势字段

