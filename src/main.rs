extern crate mpi;

use mpi::point_to_point::Status;
use mpi::traits::*;
use std::io;
use std::time::Instant;
use Plonky2_lib::aggregator::test::{
    build_ecdsa_merge_circuit, build_ecdsa_wrap_circuit, construct_ecdsa_data, deserialize_proof,
    gen_batch_ecdsa_proof, generate_ecdsa_merge_proof, get_wrapped_common,
};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    let batch_num = 5;
    let trigger: i32 = 1;
    let (
        data,
        v_msg_biguint_target,
        v_pk_x_biguint_target,
        v_pk_y_biguint_target,
        v_r_biguint_target,
        v_s_biguint_target,
    ) = construct_ecdsa_data(batch_num);

    let wrapped_circuit = build_ecdsa_wrap_circuit(&data);

    if rank == 0 {
        let merge_num: usize = (world.size() - 1).try_into().unwrap();
        let merge_circuit = build_ecdsa_merge_circuit(wrapped_circuit, merge_num);

        let wrapped_circuit_2 = build_ecdsa_wrap_circuit(&data);
        let wrapped_common = get_wrapped_common(&wrapped_circuit_2);

        let wrapped_circuit_3 = build_ecdsa_wrap_circuit(&data);
        let wrapped_data = wrapped_circuit_3.circuits_data.last().unwrap();

        let merge_circuit_2 = build_ecdsa_merge_circuit(wrapped_circuit_2, merge_num);
        let merge_data = merge_circuit_2.circuit_data;

        let mut proofs = Vec::new();
        loop {
            println!("Enter 'start' to begin collecting proofs:");
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();

            if input.trim() != "start" {
                println!("Invalid command.");
                continue;
            }

            let start = Instant::now();

            for target_rank in 1..world.size() {
                world.process_at_rank(target_rank).send(&trigger);
            }

            while proofs.len() < merge_num {
                let (proof_size, status) = world.any_process().receive::<i32>();
                let mut proof_bytes = vec![0u8; proof_size as usize];
                let source_rank = status.source_rank();
                world
                    .process_at_rank(source_rank)
                    .receive_into(&mut proof_bytes);

                let received_time = Instant::now();
                let time_since_start = received_time.duration_since(start);
                println!(
                    "Received proof from rank {} at {:?} since start, proof size: {}",
                    source_rank, time_since_start, proof_size
                );

                let proof = deserialize_proof(proof_bytes, &wrapped_common);
                wrapped_data
                    .verify(proof.clone())
                    .expect("ecdsa wrapped proof verification failed");

                println!(
                    "Verified proof from rank {} at rank {}, proof PIS: {:?}",
                    source_rank, rank, proof.public_inputs
                );

                proofs.push(proof);
            }
            let merge_proof = generate_ecdsa_merge_proof(&merge_circuit, proofs.clone()).unwrap();

            merge_data
                .verify(merge_proof.clone())
                .expect("merge proof verification failed");
            proofs.clear();

            println!("merge proof PIS: {:?}", merge_proof.public_inputs);

            let duration = start.elapsed();
            println!(
                "Total time for collecting and merging proofs: {:?}",
                duration
            );
        }
    } else {
        loop {
            let task = world.process_at_rank(0).receive::<i32>();
            let proof = gen_batch_ecdsa_proof(
                batch_num,
                &data,
                v_msg_biguint_target.clone(),
                v_pk_x_biguint_target.clone(),
                v_pk_y_biguint_target.clone(),
                v_r_biguint_target.clone(),
                v_s_biguint_target.clone(),
            );
            println!(
                "Generated proof at rank {}, proof PIS: {:?}",
                rank, proof.public_inputs
            );
            let wrappred_proof_final = wrapped_circuit
                .generate_wrap_proofs(proof)
                .unwrap()
                .last()
                .unwrap()
                .clone();

            let proof_bytes = wrappred_proof_final.to_bytes();
            let proof_size = proof_bytes.len() as i32;
            world.process_at_rank(0).send(&proof_size);
            world.process_at_rank(0).send(&proof_bytes);
        }
    }
}
