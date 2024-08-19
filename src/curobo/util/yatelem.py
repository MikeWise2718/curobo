#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Standard Library
import io
import time
import argparse
import traceback

stats = {}
msgs = []

opt_enable_logging = False
opt_print_out = False
opt_file_out = False
opt_output_results = False
opt_output_traceback = False
opt_log_file_name = "default.txt"

def logmsg(msg):
    global opt_enable_logging
    if opt_enable_logging:
        msgs.append(msg)

def writeout_log():
    global opt_enable_logging, opt_log_file_name
    if opt_enable_logging:
        nlog = len(msgs)
        print(f"Writing {nlog} log linees to", opt_log_file_name)
        with open(opt_log_file_name, "w") as f:
            for msg in msgs:
                f.write(f"{msg}\n")

def print_to_string(*args, **kwargs):
    # output = io.StringIO()
    # print(*args, file=output, **kwargs)
    # contents = output.getvalue()
    # output.close()
    newstr = ""
    # for a in args:
    #     newstr += str(a) + ' '
    return newstr

def record_start_call(caller_id):
    startcount = 0
    if caller_id not in stats:
        newdict = {}
        newdict["startcount"] = 0
        newdict["endcount"] = 0
        newdict["total_time"] = 0
        newdict["min_time"] = 1000000
        newdict["max_time"] = 0
        stats[caller_id] = newdict
    else:
        stats[caller_id]["startcount"] += 1
        startcount = stats[caller_id]["startcount"]
    return startcount

def record_end_call(caller_id, elap):
    stats[caller_id]["endcount"] += 1
    stats[caller_id]["total_time"] += elap
    stats[caller_id]["min_time"] = min(stats[caller_id]["min_time"], elap)
    stats[caller_id]["max_time"] = max(stats[caller_id]["max_time"], elap)


def start_cuda_call(caller_id):
    global opt_enable_logging, opt_print_out, opt_file_out
    if not opt_enable_logging:
        return 0
    startcount = record_start_call(caller_id)
    if startcount == 21:
        pass
    msg = f"------------------ {caller_id} -----------------sc:{startcount}"
    if opt_print_out:
        print(msg)
    if opt_file_out:
        logmsg(msg)
    start = time.time()
    cc_dict = {"caller_id":caller_id, "start": start}
    return cc_dict

def end_cuda_call(cc_dict, result=""):
    global opt_enable_logging, opt_print_out, opt_file_out, opt_output_results, opt_output_traceback
    if not opt_enable_logging:
        return
    caller_id = cc_dict["caller_id"]
    elap = time.time() - cc_dict["start"]
    msg = f"                  {caller_id} took {elap:.6f} secs"
    #  if result != "":
    #      emsg += f" with result {result}"
    if opt_print_out:
        print(msg)
    if opt_file_out:
        logmsg(msg)
        if opt_output_results:
            # resstr = print_to_string(result)
            logmsg(str(result))
        if opt_output_traceback:
            exlst = traceback.extract_stack(f=None, limit=None)
            tb = traceback.format_list(exlst)
            for tbline in tb:
                logmsg(tbline)

    record_end_call(caller_id, elap)
    return
