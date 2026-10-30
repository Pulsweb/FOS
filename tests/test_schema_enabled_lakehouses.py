import json
import re
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
MODULES_ROOT = REPO_ROOT / "src" / "modules"
ADMIN_LAKEHOUSE_TABLE_PATTERN = re.compile(
    r"\bFOS_Lakehouse\."
    r"(?!adm(?:\.|\b)|staging(?:\.|\b)|config(?:\.|\b)|"
    r"capacity_metrics(?:\.|\b)|\{schema_name\}|Lakehouse\b)"
)


class SchemaEnabledLakehouseTests(unittest.TestCase):
    def test_admin_module_declares_required_schemas(self):
        config = yaml.safe_load(
            (
                REPO_ROOT
                / "config/modules/admin-monitoring/deployment_config.yaml"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            config["lakehouse_schemas"],
            {
                "FOS_Lakehouse.Lakehouse": [
                    "adm",
                    "staging",
                    "config",
                    "capacity_metrics",
                ],
            },
        )

    def test_deployment_creates_schema_enabled_lakehouses_without_spark(self):
        notebook = json.loads(
            (REPO_ROOT / "scripts" / "Deploy_FOS.ipynb").read_text(encoding="utf-8")
        )
        source = "\n".join(
            line
            for cell in notebook["cells"]
            for line in cell.get("source", [])
        )
        self.assertIn("enableSchemas=True", source)
        self.assertNotIn("enableSchemas=False", source)
        self.assertIn("properties.defaultSchema", source)
        self.assertEqual(source.count("properties.defaultSchema"), 1)
        self.assertIn("Existing non-schema", source)
        self.assertIn("lakehouse_schemas", source)
        self.assertNotIn("spark.", source)
        self.assertNotIn("pandas", source)
        self.assertIn("notebookutils.fs.cp", source)

        admin_initializer = json.loads(
            (
                MODULES_ROOT
                / "admin-monitoring/Init_FOS_Lakehouse_Tables.Notebook/notebook-content.ipynb"
            ).read_text(encoding="utf-8")
        )
        initializer_source = "\n".join(
            line
            for cell in admin_initializer["cells"]
            for line in cell.get("source", [])
        )
        self.assertIn(
            'spark.read.parquet(table_definitions_path)',
            initializer_source,
        )

    def test_initializers_create_declared_schemas_before_use(self):
        expected_schemas = {
            "admin-monitoring/Init_FOS_Lakehouse_Tables.Notebook/notebook-content.ipynb": (
                "adm",
                "staging",
                "config",
                "capacity_metrics",
            ),
            "cost-analysis/Init_Cost_Analysis_Lakehouse_Tables.Notebook/notebook-content.ipynb": (
                "cost",
            ),
        }
        for relative_path, schemas in expected_schemas.items():
            with self.subTest(path=relative_path):
                notebook = json.loads(
                    (MODULES_ROOT / relative_path).read_text(encoding="utf-8")
                )
                source = "\n".join(
                    line
                    for cell in notebook["cells"]
                    for line in cell.get("source", [])
                )
                self.assertIn("CREATE SCHEMA IF NOT EXISTS", source)
                self.assertLess(
                    source.index("CREATE SCHEMA IF NOT EXISTS"),
                    source.index(f'USE {schemas[0]}'),
                )
                for schema in schemas:
                    self.assertIn(schema, source)

    def test_each_module_notebook_selects_its_schema_once(self):
        for path in MODULES_ROOT.rglob("*.ipynb"):
            with self.subTest(path=path):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                if "cost-analysis" in path.parts:
                    expected_schema = "cost"
                elif any("CapacityMetricData" in part for part in path.parts):
                    expected_schema = "capacity_metrics"
                else:
                    expected_schema = "adm"
                count = sum(
                    line.strip() == f'spark.sql("USE {expected_schema}")'
                    for cell in notebook["cells"]
                    for line in cell.get("source", [])
                )
                self.assertEqual(count, 1)

    def test_admin_notebook_table_references_use_declared_schemas(self):
        violations = []
        admin_root = MODULES_ROOT / "admin-monitoring"
        for path in admin_root.rglob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell.get("cell_type") != "code":
                    continue
                for line in cell.get("source", []):
                    if ADMIN_LAKEHOUSE_TABLE_PATTERN.search(line):
                        violations.append(f"{path}: {line.strip()}")
                    if '"Tables/dbo/' in line or "'Tables/dbo/" in line:
                        violations.append(f"{path}: {line.strip()}")
                    if "FOS_Staging_Lakehouse" in line:
                        violations.append(f"{path}: {line.strip()}")
        self.assertEqual(violations, [])

    def test_reserved_admin_schema_is_not_used_for_storage(self):
        storage_tokens = (
            "USE admin",
            "FOS_Lakehouse.admin",
            "Tables/admin",
            "schemaName: admin",
            "[admin].",
        )
        files = [
            *MODULES_ROOT.glob("admin-monitoring/**/*.ipynb"),
            *MODULES_ROOT.glob("admin-monitoring/**/*.tmdl"),
        ]
        for path in files:
            with self.subTest(path=path):
                text = path.read_text(encoding="utf-8")
                for token in storage_tokens:
                    self.assertNotIn(token, text)

    def test_admin_staging_tables_use_shared_lakehouse(self):
        admin_root = MODULES_ROOT / "admin-monitoring"
        serialized = "\n".join(
            path.read_text(encoding="utf-8")
            for path in admin_root.rglob("*.ipynb")
        )
        self.assertNotIn("FOS_Staging_Lakehouse", serialized)
        self.assertIn("FOS_Lakehouse.staging", serialized)

        order = json.loads(
            (
                REPO_ROOT
                / "config/modules/admin-monitoring/deployment_order.json"
            ).read_text(encoding="utf-8")
        )
        self.assertNotIn(
            "FOS_Staging_Lakehouse.Lakehouse",
            {item["name"] for item in order},
        )
        module_files = [
            REPO_ROOT / "cicd/modules/admin-monitoring/item_src_config.yaml",
            REPO_ROOT / "config/modules/admin-monitoring/deployment_config.yaml",
            REPO_ROOT / "config/modules/admin-monitoring/module.yaml",
        ]
        for path in module_files:
            with self.subTest(path=path):
                text = path.read_text(encoding="utf-8")
                for removed_name in (
                    "FOS_Staging_Lakehouse",
                    "FOS_Config_Lakehouse",
                    "FOS_Backup_Lakehouse",
                    "01_FOS_Lakehouse_Backup",
                ):
                    self.assertNotIn(removed_name, text)

    def test_removed_admin_lakehouses_and_backup_notebook_are_absent(self):
        removed_items = {
            "FOS_Staging_Lakehouse.Lakehouse",
            "FOS_Config_Lakehouse.Lakehouse",
            "FOS_Backup_Lakehouse.Lakehouse",
            "01_FOS_Lakehouse_Backup.Notebook",
        }
        order = json.loads(
            (
                REPO_ROOT
                / "config/modules/admin-monitoring/deployment_order.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(removed_items.isdisjoint({item["name"] for item in order}))
        self.assertFalse(
            (
                MODULES_ROOT
                / "admin-monitoring/01_FOS_Lakehouse_Backup.Notebook/notebook-content.ipynb"
            ).exists()
        )

    def test_metrics_tables_use_capacity_metrics_schema(self):
        metrics_tables = {
            "capacity_metrics_by_item_by_operation_by_day",
            "capacity_metrics_by_item_kind_by_day",
            "capacity_metrics_by_timepoint",
        }
        for path in MODULES_ROOT.glob(
            "admin-monitoring/*.SemanticModel/definition/tables/*metrics*.tmdl"
        ):
            text = path.read_text(encoding="utf-8")
            if any(f"entityName: {table}" in text for table in metrics_tables):
                with self.subTest(path=path):
                    self.assertIn("schemaName: capacity_metrics", text)

        init_notebook = json.loads(
            (
                MODULES_ROOT
                / "admin-monitoring/Init_FOS_Lakehouse_Tables.Notebook/notebook-content.ipynb"
            ).read_text(encoding="utf-8")
        )
        init_source = "\n".join(
            line for cell in init_notebook["cells"] for line in cell.get("source", [])
        )
        self.assertIn(
            '"capacity_metrics" if "metrics" in table_name.lower() else "adm"',
            init_source,
        )

    def test_cost_analysis_uses_cost_schema(self):
        config = yaml.safe_load(
            (
                REPO_ROOT
                / "config/modules/cost-analysis/deployment_config.yaml"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            config["lakehouse_schemas"],
            {"FOS_Lakehouse.Lakehouse": ["cost"]},
        )

        pipeline_path = (
            MODULES_ROOT
            / "cost-analysis"
            / "Load Quota.DataPipeline"
            / "pipeline-content.json"
        )
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        subscriptions_activity = next(
            activity
            for activity in pipeline["properties"]["activities"]
            if activity["name"] == "Get Subscriptions"
        )
        self.assertEqual(
            subscriptions_activity["typeProperties"]["datasetSettings"]["typeProperties"],
            {"schema": "cost", "table": "subscriptions"},
        )

    def test_direct_lake_partitions_use_module_schema(self):
        table_files = list(MODULES_ROOT.rglob("*.SemanticModel/definition/tables/*.tmdl"))
        direct_lake_files = [
            path
            for path in table_files
            if "mode: directLake" in path.read_text(encoding="utf-8")
        ]
        self.assertTrue(direct_lake_files)
        for path in direct_lake_files:
            with self.subTest(path=path):
                if "cost-analysis" in path.parts:
                    expected_schema = "cost"
                elif "metrics" in path.stem.lower():
                    expected_schema = "capacity_metrics"
                else:
                    expected_schema = "adm"
                self.assertIn(
                    f"schemaName: {expected_schema}",
                    path.read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
