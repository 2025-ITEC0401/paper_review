#!/bin/bash
nohup ./store1.sh > 0_nohup_store1.log &
nohup ./store2.sh > 0_nohup_store2.log &
