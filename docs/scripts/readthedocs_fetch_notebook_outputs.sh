#! /bin/bash
#
# Copyright 2024 The Penzai Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Attempts to load pre-rendered notebook outputs from a git tag or branch
# when building documentation with ReadTheDocs. Notebook outputs are expected
# to be found on another tag or branch with a +notebook_outputs suffix.
#
# This script parses the current git branch or tag, and uses it to determine the
# name of another branch or tag where the notebook outputs are expected to be
# found. It then fetches this ref and makes sure it is exactly one commit ahead
# of the current HEAD.
#
# If the current ref is a tag, and we don't find the notebook outputs, we
# fail the ReadTheDocs build. Tags are assumed to be named versions that should
# always contain the notebook outputs in the documentation.
#
# If the current ref is a branch, we don't require the notebook outputs to be
# found, and just render documentation without them.
#
# The goal of this approach is to allow pre-rendering notebook outputs in
# the stable version of Penzai's documentation, without having the notebook
# outputs embedded in the main branch or the actual released package.

set -e
set -x

# Figure out where we are, likely a branch or tag.
# - When building a tag, this will produce something like "tags/tagname^0"
# - When building a branch, it will be like "remotes/origin/branchname"
curref=$(git name-rev --name-only --no-undefined --exclude='main' --exclude='HEAD' HEAD || true)

if [[ "${curref}" =~ ^tags/(.*)"^0"$ ]]; then
  # Looks like a tag.
  tagname="${BASH_REMATCH[1]}"
  notebook_ref=${tagname}+notebook_outputs
  missing_message="ERROR: Missing notebook outputs ref ${notebook_ref} for tag ${tagname}"
  required="true"
elif [[ "${curref}" =~ ^remotes/origin/(.*)$ ]]; then
  # Looks like a branch.
  branchname="${BASH_REMATCH[1]}"
  notebook_ref=${branchname}+notebook_outputs
  missing_message="INFO: Missing notebook outputs ref ${notebook_ref} for branch ${branchname}"
  required="false"
else
  # Not a branch or tag.
  notebook_ref=""
  missing_message="INFO: Couldn't determine notebook outputs ref for $(git describe --always)"
  required="false"
fi

if [ -n "${notebook_ref}" ] \
  && git fetch origin "${notebook_ref}" \
  && [ "$(git rev-parse --verify HEAD)" \
       == "$(git rev-parse --verify FETCH_HEAD~)" ]
then
  # Check out the notebooks from the notebook ref.
  git checkout FETCH_HEAD .
  git reset
elif [ "${required}" != "false" ]; then
  printf "%s\n" "${missing_message}" >&2
  exit 1
else
  printf "%s\n" "${missing_message}" >&2
fi
