# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import glob
import os
import platform
import tempfile
import typing as ty
import subprocess  # noqa: S404
import unittest
from test import support
import nbformat

import lava


class TestTutorials(unittest.TestCase):
    """Export notebook, execute to check for errors."""

    system_name = platform.system().lower()

    def _execute_notebook(self, base_dir: str, path: str) -> \
            int:
        """Execute a notebook via nbconvert and collect output.

        Parameters
        ----------
        base_dir : str
            notebook search directory
        path : str
            path to notebook

        Returns
        -------
        int
            (return code)
        """

        cwd = os.getcwd()
        dir_name, notebook = os.path.split(path)
        try:
            env = self._update_pythonpath(base_dir, dir_name)
            result = self._convert_and_execute_notebook(notebook, env)
            errors = self._collect_errors_from_all_cells(result)
        except Exception as e:
            nb = None
            errors = str(e)
        finally:
            os.chdir(cwd)

        return errors

    def _update_pythonpath(self, base_dir: str, dir_name: str) \
            -> ty.Dict[str, str]:
        """Update PYTHONPATH with notebook location.

        Parameters
        ----------
        base_dir : str
            Parent directory to use
        dir_name : str
            Directory containing notebook

        Returns
        -------
        env : dict
            Updated dictionary of environment variables
        """
        os.chdir(base_dir + "/" + dir_name)

        env = os.environ.copy()
        module_path = [lava.__path__.__dict__["_path"][0]]
        # Path: module path + parent dir of module + existing PYTHONPATH
        module_path.extend(
            [os.path.dirname(module_path[0]), env.get("PYTHONPATH", "")])
        env["PYTHONPATH"] = ":".join(module_path)
        return env

    def _convert_and_execute_notebook(self, notebook: str,
                                      env: ty.Dict[str, str]) \
            -> ty.Type[nbformat.NotebookNode]:
        """Covert notebook and execute it.

        Parameters
        ----------
        notebook : str
            Notebook name
        env : dict
            Dictionary of environment variables

        Returns
        -------
        nb : nbformat.NotebookNode
            Notebook dict-like node with attribute-access
        """
        with tempfile.NamedTemporaryFile(mode="w+t", suffix=".py") \
                as fout:
            args = ["jupyter", "nbconvert", "--to", "python", "--execute",
                    "--output", fout.name, notebook]
            subprocess.check_call(args, env=env)  # noqa: S603

            fout.seek(0)
            return subprocess.run(["ipython", "-c", fout.read()], env=env)

    def _collect_errors_from_all_cells(self, result) \
            -> ty.List[str]:
        """Collect errors from executed notebook.

        Parameters
        ----------
        nb : nbformat.NotebookNode
            Notebook to search for errors

        Returns
        -------
        List
            Collection of errors
        """
        if result.returncode != 0:
            result.check_returncode()
        return result.returncode

    def _run_notebook(self, notebook: str):
        """Run a specific notebook

        Parameters
        ----------
        notebook : str
            name of notebook to run
        """

        cwd = os.getcwd()
        tutorials_temp_directory = \
            os.path.abspath(os.path.join(os.getcwd(), os.path.pardir,
                                         os.path.pardir, os.path.pardir))
        tutorials_directory = ""

        tutorials_directory = os.path.realpath(tutorials_temp_directory)
        os.chdir(tutorials_directory)

        errors_record = {}

        try:
            glob_pattern = "**/{}".format(notebook)
            discovered_notebooks = sorted(
                glob.glob(glob_pattern, recursive=True))

            self.assertTrue(len(discovered_notebooks) != 0,
                            "Notebook not found. Input to function {}"
                            .format(notebook))

            # If the notebook is found execute it and store any errors
            for notebook_name in discovered_notebooks:
                errors = self._execute_notebook(
                    str(tutorials_directory), notebook_name
                )
                self.assertEqual(errors, 0)
        finally:
            os.chdir(cwd)

    def test_tutorial_01_solving_lasso(self):
        """Test tutorial on Lava lasso solver."""
        self._run_notebook(
            "tutorial_01_solving_lasso.ipynb"
        )

    def test_tutorial_02_solving_qubos(self):
        """Test tutorial on Lava QUBO solver."""
        self._run_notebook(
            "tutorial_02_solving_qubos.ipynb"
        )


if __name__ == '__main__':
    support.run_unittest(TestTutorials)
