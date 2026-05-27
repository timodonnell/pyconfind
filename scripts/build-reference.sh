#!/usr/bin/env bash
# Build the upstream C++ confind binary used as the validation reference.
#
# Expects the upstream tarball at original-source/confind-msl.tar.gz (downloaded
# from grigoryanlab.org). Extracts the tree, applies the patches needed to make
# the 2017-era code compile against modern glibc / gcc 11+, builds, and runs a
# smoke test.
#
# The patches:
#   1. mslib/src/CartesianPoint.h — make CartesianPointCompare::operator() const
#      so it can be used as a std::map comparator under post-C++17 enforcement.
#   2. mslib/myProgs/gevorg/confind.cpp — replace an rvalue-address pattern in
#      toString<T> that gcc 11 rejects.
#   3. mslib/myProgs/gevorg/gevorg.mk — point MSL_EXTERNAL_LIB_DIR at
#      /usr/lib/x86_64-linux-gnu (Ubuntu's libgsl install location).
#
# Requires: g++, make, libgsl-dev (apt: libgsl-dev).
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
src_dir="${repo_root}/original-source"
tarball="${src_dir}/confind-msl.tar.gz"
work_dir="${src_dir}/confind-msl"
mslib="${work_dir}/mslib"

if [[ ! -f "${tarball}" ]]; then
    echo "Expected upstream tarball at ${tarball}" >&2
    exit 1
fi

if [[ ! -d "${work_dir}" ]]; then
    echo "Extracting upstream tarball..."
    tar -xzf "${tarball}" -C "${src_dir}"
fi

# Patch 1: const-qualify CartesianPointCompare::operator().
sed -i 's|bool operator()(const CartesianPoint &pt1, const CartesianPoint &pt2) {|bool operator()(const CartesianPoint &pt1, const CartesianPoint &pt2) const {|' \
    "${mslib}/src/CartesianPoint.h"

# Patch 2: rewrite toString to avoid taking the address of an rvalue.
python3 - "${mslib}/myProgs/gevorg/confind.cpp" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
text = p.read_text()
old = (
    "template <class T>\n"
    "string toString (T val) {\n"
    "  return static_cast<ostringstream*>( &(ostringstream() << val) )->str();\n"
    "}\n"
)
new = (
    "template <class T>\n"
    "string toString (T val) {\n"
    "  ostringstream oss;\n"
    "  oss << val;\n"
    "  return oss.str();\n"
    "}\n"
)
if old in text:
    p.write_text(text.replace(old, new))
PY

# Patch 3: point library/include paths at Ubuntu defaults.
sed -i \
    -e 's|MSL_EXTERNAL_LIB_DIR=/usr/local/lib|MSL_EXTERNAL_LIB_DIR=/usr/lib/x86_64-linux-gnu|' \
    -e 's|MSL_EXTERNAL_INCLUDE_DIR=/usr/local/include|MSL_EXTERNAL_INCLUDE_DIR=/usr/include|' \
    "${mslib}/myProgs/gevorg/gevorg.mk"

cd "${mslib}"
MSL_GSL=T make -j"$(nproc)" bin/confind

cd "${repo_root}"
echo
echo "Built ${mslib}/bin/confind"
echo "Smoke test:"
"${mslib}/bin/confind" --p "${mslib}/exampleFiles/example0000.pdb" \
    --rLib "${work_dir}/rotlibs" 2>&1 | head -5
