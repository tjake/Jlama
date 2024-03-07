/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.github.tjake.jlama.cli.serve;

import com.github.tjake.jlama.model.functions.Generator;
import org.jboss.resteasy.plugins.providers.jackson.ResteasyJackson2Provider;

import jakarta.ws.rs.core.Application;
import java.util.HashMap;
import java.util.HashSet;
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
