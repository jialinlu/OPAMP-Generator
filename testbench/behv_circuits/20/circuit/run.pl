#!/usr/bin/perl
use strict;
use warnings;
use 5.010;
`./clear.sh`;
`rm -f result.po`;
`python ocnScript_generate.py`;
`ocean -replay ./oceanScript_opamp.ocn -log ocean.log > err 2>&1`;
# `ocean -replay ./oceanScript_opamp.ocn -nograph -log ocean.log > err 2>&1`;
`python extract.py`;
`cat param result.po >> backup`;