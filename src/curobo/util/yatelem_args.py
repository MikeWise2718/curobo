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


parser = argparse.ArgumentParser()
def add_yatelem_args():
    parser.add_argument(
        "-log",
        action="store_true",
        help="Log telemetry",
    )
    parser.add_argument(
        "-logname",
        type=str,
        default="default.txt",
        help="File name to log results to",
    )

    parser.add_argument(
        "--no_print_log",
        action="store_true",
        help="Do not print log",
        default=False,
    )

    parser.add_argument(
        "-ar",
        "--add_results",
        action="store_true",
        help="Add results to log",
    )

    parser.add_argument(
        "-at",
        "--add_traceback",
        action="store_true",
        help="Add traceback to log",
    )


add_yatelem_args()
args = parser.parse_args()
print("parsed args:", args)
print("args.no_print_log:", args.no_print_log)


stats = {}
msgs = []


def logmsg(msg):
    global args
    if args.log:
        msgs.append(msg)

def writeout_log():
    global args
    if args.log:
        nlog = len(msgs)
        print(f"Writing {nlog} log linees to", args.logname)
        with open(args.logname, "w") as f:
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
    global args
    startcount = record_start_call(caller_id)
    msg = f"------------------ {caller_id} -----------------sc:{startcount}"
    if not args.no_print_log:
        print(msg)
    if args.log:
        logmsg(msg)
    start = time.time()
    cc_dict = {"caller_id":caller_id, "start": start}
    return cc_dict

def end_cuda_call(cc_dict, result=""):
    global args
    caller_id = cc_dict["caller_id"]
    elap = time.time() - cc_dict["start"]
    msg = f"                  {caller_id} took {elap:.6f} secs"
    #  if result != "":
    #      emsg += f" with result {result}"
    if not args.no_print_log:
        print(msg)
    if args.log:
        logmsg(msg)
        if args.add_results:
            # resstr = print_to_string(result)
            logmsg(str(result))
        if args.add_traceback:
            exlst = traceback.extract_stack(f=None, limit=None)
            tb = traceback.format_list(exlst)
            for tbline in tb:
                logmsg(tbline)

    record_end_call(caller_id, elap)
    return
