package com.github.tjake.jlama.cli.serve;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.functions.Generator;
import org.jboss.resteasy.plugins.providers.jackson.ResteasyJackson2Provider;

import jakarta.ws.rs.ApplicationPath;
import jakarta.ws.rs.core.Application;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class JlamaRestApi extends Application {

    final Generator model;
    public JlamaRestApi(Generator model) {
        this.model = model;
    }

    @Override
    public Set<Class<?>> getClasses() {
        Set<Class<?>> resources = new HashSet<>();
        resources.add(ResteasyJackson2Provider.class);
        return resources;
    }

    @Override
    public Set<Object> getSingletons() {
        Set<Object> set = new HashSet<>();
        set.add(new GenerateResource(model));
        set.add(new ChatApi(model));
        set.add(new EmbedResource());
        return set;
    }
}