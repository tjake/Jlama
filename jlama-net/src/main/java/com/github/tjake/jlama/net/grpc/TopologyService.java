package com.github.tjake.jlama.net.grpc;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import com.github.tjake.jlama.model.functions.Generator;
import com.github.tjake.jlama.net.Coordinator;
import com.github.tjake.jlama.net.RegisterResponse;
import com.github.tjake.jlama.net.openai.model.CreateChatCompletionRequest;
import com.github.tjake.jlama.safetensors.Config;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RestController
@Validated
public class TopologyService
{

    @Autowired
    private Generator model;

    /**
     * GET /coordinator/topology : Returns the current topology of the cluster.
     *
     * @return OK (status code 200)
     */
    @RequestMapping(method = RequestMethod.GET, value = "/cluster/topology", produces = { "application/json" })
    public Object getTopology()
    {
        if (!(model instanceof Coordinator)) {
            return new ResponseEntity<>(HttpStatus.BAD_GATEWAY);
        }

        Config config = ((Coordinator) model).getConfig();
        Map<UUID, RegisterResponse> workers = ((Coordinator) model).getWorkers();

        List<Map<String, String>> workerList = new ArrayList<>();
        for (Map.Entry<UUID, RegisterResponse> entry : workers.entrySet()) {
            String id = entry.getKey().toString();
            RegisterResponse w = entry.getValue();

            workerList.add(Map.of(
                    "id", id,
                    "address", w.getHostname(),
                    "layer_shard", Integer.toString(w.getLayerShard()),
                    "head_shard", Integer.toString(w.getModelShard()),
                    "layer_shard_total", Integer.toString(w.getNumLayerShards()),
                    "head_shard_total", Integer.toString(w.getNumModelShards()),
                    "ordinal", Integer.toString(w.getWorkerOrd())));
        }

        Map<String, Object> topology = Map.of(
                "num_layers", config.numberOfLayers,
                "num_heads", config.numberOfKeyValueHeads,
                "num_workers", workerList.size(),
                "workers", workerList);

        return new ResponseEntity<>(topology, HttpStatus.OK);
    }
}