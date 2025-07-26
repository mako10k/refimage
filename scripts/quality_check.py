#!/usr/bin/env python3
"""
Automated code quality check script for RefImage project.

This script runs comprehensive code quality checks including:
- Code formatting (Black)
- Import sorting (isort) 
- Linting (Flake8)
- Type checking (MyPy)
- Security scanning (Bandit)
- Code duplication detection (jscpd)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CodeQualityChecker:
    """Automated code quality checker."""

    def __init__(self, project_root: Path):
        """
        Initialize code quality checker.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root
        self.src_paths = ["src/", "tests/"]
        self.results: Dict[str, Dict] = {}

    def run_command(self, cmd: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
        """
        Run shell command and return results.
        
        Args:
            cmd: Command to execute
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                timeout=300
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)

    def check_black(self, fix: bool = False) -> Dict:
        """
        Check code formatting with Black.
        
        Args:
            fix: Whether to automatically fix issues
            
        Returns:
            Results dictionary
        """
        print("🔍 Checking code formatting (Black)...")
        
        cmd = ["black"]
        if not fix:
            cmd.extend(["--check", "--diff"])
        cmd.extend(self.src_paths)
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        result = {
            "tool": "black",
            "passed": exit_code == 0,
            "exit_code": exit_code,
            "output": stdout,
            "errors": stderr
        }
        
        if fix and exit_code == 0:
            print("✅ Black: Code formatting fixed")
        elif exit_code == 0:
            print("✅ Black: Code formatting is correct")
        else:
            print("❌ Black: Code formatting issues found")
            
        return result

    def check_isort(self, fix: bool = False) -> Dict:
        """
        Check import sorting with isort.
        
        Args:
            fix: Whether to automatically fix issues
            
        Returns:
            Results dictionary
        """
        print("🔍 Checking import sorting (isort)...")
        
        cmd = ["isort"]
        if not fix:
            cmd.extend(["--check-only", "--diff"])
        cmd.extend(self.src_paths)
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        result = {
            "tool": "isort",
            "passed": exit_code == 0,
            "exit_code": exit_code,
            "output": stdout,
            "errors": stderr
        }
        
        if fix and exit_code == 0:
            print("✅ isort: Import sorting fixed")
        elif exit_code == 0:
            print("✅ isort: Import sorting is correct")
        else:
            print("❌ isort: Import sorting issues found")
            
        return result

    def check_flake8(self) -> Dict:
        """
        Check code style and syntax with Flake8.
        
        Returns:
            Results dictionary
        """
        print("🔍 Checking code style (Flake8)...")
        
        cmd = [
            "flake8",
            "--max-line-length=88",
            "--ignore=E203,W503,E501",  # Compatible with Black
            "--exclude=__pycache__,.git,.venv,venv"
        ]
        cmd.extend(self.src_paths)
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        result = {
            "tool": "flake8",
            "passed": exit_code == 0,
            "exit_code": exit_code,
            "output": stdout,
            "errors": stderr,
            "issues_count": len(stdout.split('\n')) - 1 if stdout.strip() else 0
        }
        
        if exit_code == 0:
            print("✅ Flake8: No code style issues found")
        else:
            print(f"❌ Flake8: {result['issues_count']} code style issues found")
            
        return result

    def check_mypy(self) -> Dict:
        """
        Check type annotations with MyPy.
        
        Returns:
            Results dictionary
        """
        print("🔍 Checking type annotations (MyPy)...")
        
        cmd = [
            "mypy",
            "src/refimage",
            "--ignore-missing-imports",
            "--no-strict-optional",
            "--warn-return-any",
            "--warn-unused-configs"
        ]
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        result = {
            "tool": "mypy",
            "passed": exit_code == 0,
            "exit_code": exit_code,
            "output": stdout,
            "errors": stderr,
            "issues_count": len(stdout.split('\n')) - 1 if stdout.strip() else 0
        }
        
        if exit_code == 0:
            print("✅ MyPy: No type issues found")
        else:
            print(f"❌ MyPy: {result['issues_count']} type issues found")
            
        return result

    def check_bandit(self) -> Dict:
        """
        Check security issues with Bandit.
        
        Returns:
            Results dictionary
        """
        print("🔍 Checking security issues (Bandit)...")
        
        cmd = [
            "bandit",
            "-r", "src/",
            "-f", "json",
            "--severity-level", "medium"
        ]
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        # Parse JSON output
        security_data = {}
        if stdout:
            try:
                security_data = json.loads(stdout)
            except json.JSONDecodeError:
                pass
        
        metrics = security_data.get('metrics', {}).get('_totals', {})
        high_severity = metrics.get('SEVERITY.HIGH', 0)
        medium_severity = metrics.get('SEVERITY.MEDIUM', 0)
        
        result = {
            "tool": "bandit",
            "passed": high_severity == 0 and medium_severity == 0,
            "exit_code": exit_code,
            "output": stdout,
            "errors": stderr,
            "high_severity": high_severity,
            "medium_severity": medium_severity,
            "total_issues": high_severity + medium_severity
        }
        
        if result["passed"]:
            print("✅ Bandit: No security issues found")
        else:
            print(f"❌ Bandit: {result['total_issues']} security issues found "
                  f"(High: {high_severity}, Medium: {medium_severity})")
            
        return result

    def check_jscpd(self) -> Dict:
        """
        Check code duplication with jscpd.
        
        Returns:
            Results dictionary
        """
        print("🔍 Checking code duplication (jscpd)...")
        
        cmd = [
            "jscpd",
            "--min-lines", "5",
            "--min-tokens", "50",
            "--reporters", "json",
            "--output", "./jscpd-report"
        ]
        cmd.extend(self.src_paths)
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        # Read jscpd report
        duplication_data = {}
        report_file = self.project_root / "jscpd-report" / "jscpd-report.json"
        if report_file.exists():
            try:
                with open(report_file) as f:
                    duplication_data = json.load(f)
            except json.JSONDecodeError:
                pass
        
        statistics = duplication_data.get('statistics', {})
        total_lines = statistics.get('total', {}).get('lines', 0)
        duplicated_lines = statistics.get('total', {}).get('duplicatedLines', 0)
        duplication_percentage = (duplicated_lines / total_lines * 100) if total_lines > 0 else 0
        
        result = {
            "tool": "jscpd",
            "passed": duplication_percentage < 1.0,  # Less than 1% duplication
            "exit_code": exit_code,
            "output": stdout,
            "errors": stderr,
            "total_lines": total_lines,
            "duplicated_lines": duplicated_lines,
            "duplication_percentage": duplication_percentage
        }
        
        if result["passed"]:
            print(f"✅ jscpd: Low code duplication ({duplication_percentage:.2f}%)")
        else:
            print(f"❌ jscpd: High code duplication ({duplication_percentage:.2f}%)")
            
        return result

    def run_all_checks(self, fix: bool = False) -> Dict:
        """
        Run all code quality checks.
        
        Args:
            fix: Whether to automatically fix issues where possible
            
        Returns:
            Complete results dictionary
        """
        print("🚀 Starting automated code quality checks...\n")
        
        # Run checks in order
        self.results["black"] = self.check_black(fix=fix)
        self.results["isort"] = self.check_isort(fix=fix)
        self.results["flake8"] = self.check_flake8()
        self.results["mypy"] = self.check_mypy()
        self.results["bandit"] = self.check_bandit()
        self.results["jscpd"] = self.check_jscpd()
        
        return self.results

    def generate_summary(self) -> None:
        """Generate and print summary of all checks."""
        print("\n" + "="*60)
        print("📊 CODE QUALITY SUMMARY")
        print("="*60)
        
        total_checks = len(self.results)
        passed_checks = sum(1 for result in self.results.values() if result["passed"])
        
        print(f"Overall: {passed_checks}/{total_checks} checks passed")
        print()
        
        for tool, result in self.results.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"{tool.upper():>8}: {status}")
            
            # Tool-specific details
            if tool == "flake8" and not result["passed"]:
                print(f"         Issues: {result.get('issues_count', 0)}")
            elif tool == "mypy" and not result["passed"]:
                print(f"         Issues: {result.get('issues_count', 0)}")
            elif tool == "bandit" and not result["passed"]:
                print(f"         High: {result.get('high_severity', 0)}, "
                      f"Medium: {result.get('medium_severity', 0)}")
            elif tool == "jscpd":
                print(f"         Duplication: {result.get('duplication_percentage', 0):.2f}%")
        
        print()
        if passed_checks == total_checks:
            print("🎉 All code quality checks passed!")
        else:
            print(f"⚠️  {total_checks - passed_checks} checks failed. Review and fix issues.")
        
        return passed_checks == total_checks

    def save_results(self, output_file: Optional[Path] = None) -> None:
        """
        Save results to JSON file.
        
        Args:
            output_file: Output file path, defaults to code-quality-report.json
        """
        if output_file is None:
            output_file = self.project_root / "code-quality-report.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📄 Results saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Automated code quality checker")
    parser.add_argument(
        "--fix", 
        action="store_true",
        help="Automatically fix issues where possible (Black, isort)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (JSON format)"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    # Initialize checker
    checker = CodeQualityChecker(args.project_root)
    
    # Run all checks
    results = checker.run_all_checks(fix=args.fix)
    
    # Generate summary
    all_passed = checker.generate_summary()
    
    # Save results
    checker.save_results(args.output)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
