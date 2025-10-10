# SPDX-FileCopyrightText: 2024-present Brandon Rose <rose.brandon.m@gmail.com>
#
# SPDX-License-Identifier: MIT

import logging
import os

# Define a package-level logging config for bdf_pz
package_logger = logging.getLogger(__name__)
package_logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
package_logger.addHandler(handler)