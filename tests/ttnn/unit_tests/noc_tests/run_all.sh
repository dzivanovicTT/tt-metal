set -e
#tt_metal/tools/profiler/profile_this.py --collect-noc-traces -c 'pytest /localdev/sohaibnadeem/tt-metal/tests/ttnn/unit_tests/test_memory_config.py'

test_dir="/localdev/sohaibnadeem/tt-metal/tests/ttnn/unit_tests/noc_tests"
metal=/localdev/sohaibnadeem/tt-metal
npe=/localdev/sohaibnadeem/tt-npe

rm -rf $test_dir/npe_results && mkdir -p $test_dir/npe_results
#mkdir -p $test_dir/trace_results

echo "Opname,Op ID,NoC Util,DRAM BW Util,Cong Impact,% Overall Cycles" > results.csv

files=$(ls -p $test_dir | grep -v "/" | grep py)
for file in $files; do
    # get traces
    truncated_filename=$(echo $file | sed  "s/.py//")
    echo $truncated_filename
    #pytest $test_dir/$file
    #$metal/tt_metal/tools/profiler/profile_this.py --collect-noc-traces -c "pytest $test_dir/$file" -o "$test_dir/trace_results/$truncated_filename"
    rm -rf "$metal/generated/"
    TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 pytest $test_dir/$file
    #/localdev/sohaibnadeem/tt-npe/tt_npe/py/pycli/tt_npe.py -t -w some/dir/trace.json --cong-model none
    $npe/install/bin/npe_analyze_noc_trace_dir.py "$metal/generated/profiler/.logs/" -s --device_name blackhole -m 1000 -s > $test_dir/npe_results/${truncated_filename}.txt
    cat $test_dir/npe_results/${truncated_filename}.txt | grep "ID[0-9]" | sed -E "s/ID[0-9]+/$truncated_filename/" | sed -E "s/ +/,/g"  >> results.txt
done


# bad:
# tracy does not let you collect traces without running npe as well????
