#!/bin/bash
# This is only intended to be used with an editable installation of beaker.
# On non-editable installations, the build tools and UI source code will not be present.

set -eoux pipefail

BEAKER_LOCATION=$(python -c "import beaker_kernel; import os; print(os.path.dirname(beaker_kernel.__file__))")
# Traverse up to top-level of package, where the Makefile lives.
BEAKER_LOCATION="${BEAKER_LOCATION}/../"

cd $BEAKER_LOCATION
make beaker_kernel/service/ui/index.html

# Delete node_modules; very large and no longer needed once bundle is generated.
rm -rf beaker-vue/node_modules
rm -rf beaker-ts/node_modules