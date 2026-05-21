package com.example.backend_spring.Repository;

import com.example.backend_spring.Entity.DetectionRequestEntity;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Collection;
import java.util.List;

public interface DetectionRequestRepository extends JpaRepository<DetectionRequestEntity, Long> {
    List<DetectionRequestEntity> findByIsReportedTrue();

    List<DetectionRequestEntity> findByStatusIn(Collection<String> statuses);
}
