package com.example.backend_spring.Config;

import io.netty.channel.ChannelOption;
import io.netty.handler.timeout.ReadTimeoutHandler;
import io.netty.handler.timeout.WriteTimeoutHandler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.http.client.HttpClient;

import java.time.Duration;
import java.util.concurrent.TimeUnit;

@Configuration
public class WebClientConfig {

        @Bean
        public WebClient pythonWebClient() {
                // Base64 히트맵 이미지를 전송받기 위해 버퍼 메모리를 10MB로 확장
                ExchangeStrategies strategies = ExchangeStrategies.builder()
                                .codecs(configurer -> configurer.defaultCodecs().maxInMemorySize(10 * 1024 * 1024))
                                .build();

                HttpClient httpClient = HttpClient.create()
                                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 5000)
                                .responseTimeout(Duration.ofSeconds(10))
                                .doOnConnected(conn -> conn
                                                .addHandlerLast(new ReadTimeoutHandler(10, TimeUnit.SECONDS))
                                                .addHandlerLast(new WriteTimeoutHandler(10, TimeUnit.SECONDS)));

                return WebClient.builder()
                                .baseUrl("http://localhost:8000")
                                .exchangeStrategies(strategies)
                                .clientConnector(new ReactorClientHttpConnector(httpClient))
                                .build();
        }
}