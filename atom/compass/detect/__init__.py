# SPDX-License-Identifier: MIT
"""Static checks over the simulated path's source, run in CI.

Each module here parses source files and returns an exit code beside a report;
none of them runs inside a simulated run. Import the module you need; this
package re-exports nothing.
"""
