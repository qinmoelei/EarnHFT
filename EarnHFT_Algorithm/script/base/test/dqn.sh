function test_BTCTUSD() {
    counter=0
    for element in $(ls $1 | sort -t '_' -k2 -n); do
        if [ "$(basename $element)" = "log" ]; then
            continue
        fi
        epoch=$(basename $element | cut -d'_' -f2)
        if [ $((counter % 4)) -eq 0 ]; then
            cuda_number=0
        elif [ $((counter % 4)) -eq 1 ]; then
            cuda_number=1
        elif [ $((counter % 4)) -eq 2 ]; then
            cuda_number=2
        else
            cuda_number=3
        fi
        echo $counter
        echo $epoch
        echo $element

        target_path=$1"/"$element
        log_filename="log/base/BTCTUSD/test_dqn_$element.log"
        CUDA_VISIBLE_DEVICES=$cuda_number nohup python RL/agent/base/dqn_test.py \
            --test_path $target_path --dataset_name BTCTUSD \
            --valid_data_path data/BTCTUSD/valid.feather --test_data_path data/BTCTUSD/test.feather \
            --transcation_cost 0 --max_holding_number 0.01 \
            >$log_filename 2>&1 &
        last_pid=$!
        counter=$((counter + 1))
        if [ $counter -eq 50 ]; then
            break
        fi
    done
    wait $last_pid
}


function test_BTCUSDT() {
    counter=0
    for element in $(ls $1 | sort -t '_' -k2 -n); do
        if [ "$(basename $element)" = "log" ]; then
            continue
        fi
        epoch=$(basename $element | cut -d'_' -f2)
        if [ $((counter % 4)) -eq 0 ]; then
            cuda_number=0
        elif [ $((counter % 4)) -eq 1 ]; then
            cuda_number=1
        elif [ $((counter % 4)) -eq 2 ]; then
            cuda_number=2
        else
            cuda_number=3
        fi
        echo $counter
        echo $epoch
        echo $element

        target_path=$1"/"$element
        log_filename="log/base/BTCUSDT/test_dqn_$element.log"
        CUDA_VISIBLE_DEVICES=$cuda_number nohup python RL/agent/base/dqn_test.py \
            --test_path $target_path --dataset_name BTCUSDT \
            --valid_data_path data/BTCUSDT/valid.feather --test_data_path data/BTCUSDT/test.feather \
            --transcation_cost 0.00015 --max_holding_number 0.01 \
            >$log_filename 2>&1 &
        last_pid=$!
        counter=$((counter + 1))
        if [ $counter -eq 50 ]; then
            break
        fi
    done
    wait $last_pid
}


function test_ETHUSDT() {
    counter=0
    for element in $(ls $1 | sort -t '_' -k2 -n); do
        if [ "$(basename $element)" = "log" ]; then
            continue
        fi
        epoch=$(basename $element | cut -d'_' -f2)
        if [ $((counter % 4)) -eq 0 ]; then
            cuda_number=0
        elif [ $((counter % 4)) -eq 1 ]; then
            cuda_number=1
        elif [ $((counter % 4)) -eq 2 ]; then
            cuda_number=2
        else
            cuda_number=3
        fi
        echo $counter
        echo $epoch
        echo $element

        target_path=$1"/"$element
        log_filename="log/base/ETHUSDT/test_dqn_$element.log"
        CUDA_VISIBLE_DEVICES=$cuda_number nohup python RL/agent/base/dqn_test.py \
            --test_path $target_path --dataset_name ETHUSDT \
            --valid_data_path data/ETHUSDT/valid.feather --test_data_path data/ETHUSDT/test.feather \
            --transcation_cost 0.00015 --max_holding_number 0.1 \
            >$log_filename 2>&1 &
        last_pid=$!
        counter=$((counter + 1))
        if [ $counter -eq 50 ]; then
            break
        fi
    done
    wait $last_pid
}



function test_GALAUSDT() {
    counter=0
    for element in $(ls $1 | sort -t '_' -k2 -n); do
        if [ "$(basename $element)" = "log" ]; then
            continue
        fi
        epoch=$(basename $element | cut -d'_' -f2)
        if [ $((counter % 4)) -eq 0 ]; then
            cuda_number=0
        elif [ $((counter % 4)) -eq 1 ]; then
            cuda_number=1
        elif [ $((counter % 4)) -eq 2 ]; then
            cuda_number=2
        else
            cuda_number=3
        fi
        echo $counter
        echo $epoch
        echo $element

        target_path=$1"/"$element
        log_filename="log/base/GALAUSDT/test_dqn_$element.log"
        CUDA_VISIBLE_DEVICES=$cuda_number nohup python RL/agent/base/dqn_test.py \
            --test_path $target_path --dataset_name GALAUSDT \
            --valid_data_path data/GALAUSDT/valid.feather --test_data_path data/GALAUSDT/test.feather \
            --transcation_cost 0.00015 --max_holding_number 4000 \
            >$log_filename 2>&1 &
        last_pid=$!
        counter=$((counter + 1))
        if [ $counter -eq 50 ]; then
            break
        fi
    done
    wait $last_pid
}

BTCTUSD_dir="result_risk/BTCTUSD/dqn_ada_0.0/seed_12345"
BTCUSDT_dir="result_risk/BTCUSDT/dqn_ada_0.0/seed_12345"
ETHUSDT_dir="result_risk/ETHUSDT/dqn_ada_0.0/seed_12345"
GALAUSDT_dir="result_risk/GALAUSDT/dqn_ada_0.0/seed_12345"




test_BTCTUSD $BTCTUSD_dir
test_BTCUSDT $BTCUSDT_dir
test_ETHUSDT $ETHUSDT_dir
test_GALAUSDT $GALAUSDT_dir










function test_BTCTUSD() {
    counter=0
    for element in $(ls $1 | sort -t '_' -k2 -n); do
        if [ "$(basename $element)" = "log" ]; then
            continue
        fi
        epoch=$(basename $element | cut -d'_' -f2)
        if [ $((counter % 4)) -eq 0 ]; then
            cuda_number=0
        elif [ $((counter % 4)) -eq 1 ]; then
            cuda_number=1
        elif [ $((counter % 4)) -eq 2 ]; then
            cuda_number=2
        else
            cuda_number=3
        fi
        echo $counter
        echo $epoch
        echo $element

        target_path=$1"/"$element
        log_filename="log/base/BTCTUSD/test_dqn_ada_256_$element.log"
        CUDA_VISIBLE_DEVICES=$cuda_number nohup python RL/agent/base/dqn_test.py \
            --test_path $target_path --dataset_name BTCTUSD \
            --valid_data_path data/BTCTUSD/valid.feather --test_data_path data/BTCTUSD/test.feather \
            --transcation_cost 0 --max_holding_number 0.01 \
            >$log_filename 2>&1 &
        last_pid=$!
        counter=$((counter + 1))
        if [ $counter -eq 50 ]; then
            break
        fi
    done
    wait $last_pid
}


function test_BTCUSDT() {
    counter=0
    for element in $(ls $1 | sort -t '_' -k2 -n); do
        if [ "$(basename $element)" = "log" ]; then
            continue
        fi
        epoch=$(basename $element | cut -d'_' -f2)
        if [ $((counter % 4)) -eq 0 ]; then
            cuda_number=0
        elif [ $((counter % 4)) -eq 1 ]; then
            cuda_number=1
        elif [ $((counter % 4)) -eq 2 ]; then
            cuda_number=2
        else
            cuda_number=3
        fi
        echo $counter
        echo $epoch
        echo $element

        target_path=$1"/"$element
        log_filename="log/base/BTCUSDT/test_dqn_ada_256_$element.log"
        CUDA_VISIBLE_DEVICES=$cuda_number nohup python RL/agent/base/dqn_test.py \
            --test_path $target_path --dataset_name BTCUSDT \
            --valid_data_path data/BTCUSDT/valid.feather --test_data_path data/BTCUSDT/test.feather \
            --transcation_cost 0.00015 --max_holding_number 0.01 \
            >$log_filename 2>&1 &
        last_pid=$!
        counter=$((counter + 1))
        if [ $counter -eq 50 ]; then
            break
        fi
    done
    wait $last_pid
}


function test_ETHUSDT() {
    counter=0
    for element in $(ls $1 | sort -t '_' -k2 -n); do
        if [ "$(basename $element)" = "log" ]; then
            continue
        fi
        epoch=$(basename $element | cut -d'_' -f2)
        if [ $((counter % 4)) -eq 0 ]; then
            cuda_number=0
        elif [ $((counter % 4)) -eq 1 ]; then
            cuda_number=1
        elif [ $((counter % 4)) -eq 2 ]; then
            cuda_number=2
        else
            cuda_number=3
        fi
        echo $counter
        echo $epoch
        echo $element

        target_path=$1"/"$element
        log_filename="log/base/ETHUSDT/test_dqn_ada_256_$element.log"
        CUDA_VISIBLE_DEVICES=$cuda_number nohup python RL/agent/base/dqn_test.py \
            --test_path $target_path --dataset_name ETHUSDT \
            --valid_data_path data/ETHUSDT/valid.feather --test_data_path data/ETHUSDT/test.feather \
            --transcation_cost 0.00015 --max_holding_number 0.1 \
            >$log_filename 2>&1 &
        last_pid=$!
        counter=$((counter + 1))
        if [ $counter -eq 50 ]; then
            break
        fi
    done
    wait $last_pid
}



function test_GALAUSDT() {
    counter=0
    for element in $(ls $1 | sort -t '_' -k2 -n); do
        if [ "$(basename $element)" = "log" ]; then
            continue
        fi
        epoch=$(basename $element | cut -d'_' -f2)
        if [ $((counter % 4)) -eq 0 ]; then
            cuda_number=0
        elif [ $((counter % 4)) -eq 1 ]; then
            cuda_number=1
        elif [ $((counter % 4)) -eq 2 ]; then
            cuda_number=2
        else
            cuda_number=3
        fi
        echo $counter
        echo $epoch
        echo $element

        target_path=$1"/"$element
        log_filename="log/base/GALAUSDT/test_dqn_ada_256_$element.log"
        CUDA_VISIBLE_DEVICES=$cuda_number nohup python RL/agent/base/dqn_test.py \
            --test_path $target_path --dataset_name GALAUSDT \
            --valid_data_path data/GALAUSDT/valid.feather --test_data_path data/GALAUSDT/test.feather \
            --transcation_cost 0.00015 --max_holding_number 4000 \
            >$log_filename 2>&1 &
        last_pid=$!
        counter=$((counter + 1))
        if [ $counter -eq 50 ]; then
            break
        fi
    done
    wait $last_pid
}








BTCTUSD_dir="result_risk/BTCTUSD/dqn_ada_256.0/seed_12345"
BTCUSDT_dir="result_risk/BTCUSDT/dqn_ada_256.0/seed_12345"
ETHUSDT_dir="result_risk/ETHUSDT/dqn_ada_256.0/seed_12345"
GALAUSDT_dir="result_risk/GALAUSDT/dqn_ada_256.0/seed_12345"




test_BTCTUSD $BTCTUSD_dir
test_BTCUSDT $BTCUSDT_dir
test_ETHUSDT $ETHUSDT_dir
test_GALAUSDT $GALAUSDT_dir