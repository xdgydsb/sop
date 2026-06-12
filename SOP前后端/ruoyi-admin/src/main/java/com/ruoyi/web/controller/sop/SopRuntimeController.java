package com.ruoyi.web.controller.sop;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.ruoyi.common.annotation.Anonymous;
import com.ruoyi.common.core.controller.BaseController;
import com.ruoyi.common.core.domain.AjaxResult;
import com.ruoyi.system.domain.sop.SopAlarmRecord;
import com.ruoyi.system.domain.sop.SopDetectionEvent;
import com.ruoyi.system.domain.sop.SopDetectionTask;
import com.ruoyi.system.domain.sop.SopProcess;
import com.ruoyi.system.domain.sop.SopProduct;
import com.ruoyi.system.domain.sop.SopRuntimeSessionRequest;
import com.ruoyi.system.domain.sop.SopRuntimeStepUpdate;
import com.ruoyi.system.domain.sop.SopRuntimeSyncRequest;
import com.ruoyi.system.domain.sop.SopRuntimeView;
import com.ruoyi.system.domain.sop.SopStep;
import com.ruoyi.system.domain.sop.SopTaskStep;
import com.ruoyi.system.mapper.sop.SopAlarmRecordMapper;
import com.ruoyi.system.mapper.sop.SopDetectionEventMapper;
import com.ruoyi.system.mapper.sop.SopDetectionTaskMapper;
import com.ruoyi.system.mapper.sop.SopProcessMapper;
import com.ruoyi.system.mapper.sop.SopProductMapper;
import com.ruoyi.system.mapper.sop.SopStepMapper;
import com.ruoyi.system.mapper.sop.SopTaskStepMapper;
import com.ruoyi.system.service.sop.ISopDetectionEventService;

@RestController
@RequestMapping("/sop/runtime")
public class SopRuntimeController extends BaseController
{
    @Autowired
    private SopDetectionTaskMapper taskMapper;

    @Autowired
    private SopTaskStepMapper taskStepMapper;

    @Autowired
    private SopStepMapper stepMapper;

    @Autowired
    private SopProcessMapper processMapper;

    @Autowired
    private SopProductMapper productMapper;

    @Autowired
    private SopDetectionEventMapper eventMapper;

    @Autowired
    private SopAlarmRecordMapper alarmMapper;

    @Autowired
    private ISopDetectionEventService detectionEventService;

    @PreAuthorize("@ss.hasPermi('sop:task:query')")
    @GetMapping("/task/{taskCode}")
    public AjaxResult getTaskRuntime(@PathVariable String taskCode)
    {
        SopDetectionTask task = taskMapper.selectSopDetectionTaskByTaskCode(taskCode);
        if (task == null)
        {
            return AjaxResult.error("任务不存在: " + taskCode);
        }
        return AjaxResult.success(buildRuntimeView(task));
    }

    @Anonymous
    @GetMapping("/current")
    public AjaxResult getCurrentRuntime(Long productId, Long sopId)
    {
        SopDetectionTask task = findCurrentTask(productId, sopId);
        return AjaxResult.success(task == null ? null : buildRuntimeView(task));
    }

    @Anonymous
    @PostMapping("/session/start")
    @Transactional
    public AjaxResult startRuntimeSession(@RequestBody SopRuntimeSessionRequest request)
    {
        if (request == null || request.getProductId() == null || request.getSopId() == null)
        {
            return AjaxResult.error("productId and sopId are required");
        }
        SopDetectionTask task = resolveSessionTask(request);
        if (task == null || StringUtils.equalsAnyIgnoreCase(task.getTaskStatus(), "PASSED", "FAILED", "FINISHED", "CANCELLED"))
        {
            cancelActiveTasks(request.getProductId(), request.getSopId(), "session restarted", request.getTaskCode());
            task = createTask(request);
        }
        armTask(task);
        return AjaxResult.success(buildRuntimeView(taskMapper.selectSopDetectionTaskByTaskId(task.getTaskId())));
    }

    @Anonymous
    @PostMapping("/session/reset")
    @Transactional
    public AjaxResult resetRuntimeSession(@RequestBody SopRuntimeSessionRequest request)
    {
        if (request == null || request.getProductId() == null || request.getSopId() == null)
        {
            return AjaxResult.error("productId and sopId are required");
        }
        cancelActiveTasks(request.getProductId(), request.getSopId(), "session reset", request.getTaskCode());
        SopDetectionTask task = createTask(request);
        return AjaxResult.success(buildRuntimeView(taskMapper.selectSopDetectionTaskByTaskId(task.getTaskId())));
    }

    @Anonymous
    @PostMapping("/session/stop")
    @Transactional
    public AjaxResult stopRuntimeSession(@RequestBody SopRuntimeSessionRequest request)
    {
        if (request == null || request.getProductId() == null || request.getSopId() == null)
        {
            return AjaxResult.error("productId and sopId are required");
        }
        SopDetectionTask task = resolveSessionTask(request);
        if (task == null)
        {
            return AjaxResult.error("no runtime session found");
        }
        stopTask(task);
        return AjaxResult.success(buildRuntimeView(taskMapper.selectSopDetectionTaskByTaskId(task.getTaskId())));
    }

    @Anonymous
    @PostMapping("/sync")
    @Transactional
    public AjaxResult syncRuntime(@RequestBody SopRuntimeSyncRequest request)
    {
        if (request == null || StringUtils.isBlank(request.getTaskCode()))
        {
            return AjaxResult.error("taskCode不能为空");
        }

        Date now = new Date();
        SopDetectionTask task = taskMapper.selectSopDetectionTaskByTaskCode(request.getTaskCode());
        if (task == null)
        {
            task = new SopDetectionTask();
            task.setTaskCode(request.getTaskCode());
            task.setProductId(request.getProductId());
            task.setSopId(request.getSopId());
            task.setStationCode(request.getStationCode());
            task.setCameraCode(request.getCameraCode());
            task.setCurrentStepNo(request.getCurrentStepNo() == null ? 1 : request.getCurrentStepNo());
            task.setTaskStatus(StringUtils.defaultIfBlank(request.getTaskStatus(), "CREATED"));
            task.setOperatorName(request.getOperatorName());
            task.setPreviewStreamUrl(request.getPreviewStreamUrl());
            task.setLatestFrameUrl(request.getLatestFrameUrl());
            task.setRuntimeMode(request.getRuntimeMode());
            task.setRuntimeMessage(request.getRuntimeMessage());
            task.setRuntimeFps(request.getRuntimeFps());
            task.setCreateBy("runtime");
            if (StringUtils.equalsAny(task.getTaskStatus(), "RUNNING", "PASSED", "FAILED", "FINISHED"))
            {
                task.setStartTime(now);
            }
            taskMapper.insertSopDetectionTask(task);
        }

        copyTaskRuntime(task, request, now);
        taskMapper.updateSopDetectionTask(task);

        List<SopStep> sopSteps = fetchSopSteps(task.getSopId());
        List<SopTaskStep> taskSteps = ensureTaskSteps(task, request.getSteps(), sopSteps);
        mergeTaskSteps(taskSteps, request.getSteps());

        SopDetectionEvent event = request.getEvent();
        SopDetectionEvent receivedEvent = null;
        if (event != null && StringUtils.isNotBlank(event.getEventCode()))
        {
            event.setTaskCode(task.getTaskCode());
            if (task.getProductCode() != null && StringUtils.isBlank(event.getProductCode()))
            {
                event.setProductCode(task.getProductCode());
            }
            if (StringUtils.isBlank(event.getStationCode()))
            {
                event.setStationCode(task.getStationCode());
            }
            if (StringUtils.isBlank(event.getCameraCode()))
            {
                event.setCameraCode(task.getCameraCode());
            }
            receivedEvent = detectionEventService.receiveVisualEvent(event);
        }

        if (receivedEvent != null && receivedEvent.getStepNo() != null)
        {
            taskSteps = fetchTaskSteps(task.getTaskId());
            final Integer matchedStepNo = receivedEvent.getStepNo();
            SopTaskStep matched = taskSteps.stream()
                .filter(item -> Objects.equals(item.getStepNo(), matchedStepNo))
                .findFirst()
                .orElse(null);
            if (matched != null)
            {
                matched.setJudgeResult(receivedEvent.getJudgeResult());
                matched.setJudgeMessage(receivedEvent.getJudgeMessage());
                matched.setSnapshotUrl(receivedEvent.getImageUrl());
                if ("PASS".equals(receivedEvent.getJudgeResult()))
                {
                    matched.setPassTime(receivedEvent.getEventTime() == null ? now : receivedEvent.getEventTime());
                }
                matched.setUpdateBy("runtime");
                taskStepMapper.updateSopTaskStep(matched);
            }
        }

        return AjaxResult.success(buildRuntimeView(taskMapper.selectSopDetectionTaskByTaskCode(task.getTaskCode())));
    }

    private void copyTaskRuntime(SopDetectionTask task, SopRuntimeSyncRequest request, Date now)
    {
        if (request.getProductId() != null)
        {
            task.setProductId(request.getProductId());
        }
        if (request.getSopId() != null)
        {
            task.setSopId(request.getSopId());
        }
        if (StringUtils.isNotBlank(request.getStationCode()))
        {
            task.setStationCode(request.getStationCode());
        }
        if (StringUtils.isNotBlank(request.getCameraCode()))
        {
            task.setCameraCode(request.getCameraCode());
        }
        if (request.getCurrentStepNo() != null)
        {
            task.setCurrentStepNo(request.getCurrentStepNo());
        }
        if (StringUtils.isNotBlank(request.getTaskStatus()))
        {
            task.setTaskStatus(request.getTaskStatus());
        }
        if (StringUtils.isNotBlank(request.getOperatorName()))
        {
            task.setOperatorName(request.getOperatorName());
        }
        if (StringUtils.isNotBlank(request.getPreviewStreamUrl()))
        {
            task.setPreviewStreamUrl(request.getPreviewStreamUrl());
        }
        if (StringUtils.isNotBlank(request.getLatestFrameUrl()))
        {
            task.setLatestFrameUrl(request.getLatestFrameUrl());
        }
        if (StringUtils.isNotBlank(request.getRuntimeMode()))
        {
            task.setRuntimeMode(request.getRuntimeMode());
        }
        if (request.getRuntimeMessage() != null)
        {
            task.setRuntimeMessage(request.getRuntimeMessage());
        }
        if (request.getRuntimeFps() != null)
        {
            task.setRuntimeFps(request.getRuntimeFps());
        }
        if (task.getStartTime() == null && StringUtils.equalsAny(task.getTaskStatus(), "RUNNING", "PASSED", "FAILED", "FINISHED"))
        {
            task.setStartTime(now);
        }
        if (StringUtils.equalsAny(task.getTaskStatus(), "PASSED", "FAILED", "FINISHED") && task.getEndTime() == null)
        {
            task.setEndTime(now);
        }
        task.setUpdateBy("runtime");
    }

    private List<SopTaskStep> ensureTaskSteps(SopDetectionTask task, List<SopRuntimeStepUpdate> runtimeSteps, List<SopStep> sopSteps)
    {
        List<SopTaskStep> existing = fetchTaskSteps(task.getTaskId());
        if (!existing.isEmpty())
        {
            return existing;
        }

        List<SopTaskStep> created = new ArrayList<>();
        if (runtimeSteps != null && !runtimeSteps.isEmpty())
        {
            Map<Integer, SopStep> sopStepByNo = new HashMap<>();
            Map<String, SopStep> sopStepByEvent = new HashMap<>();
            for (SopStep sopStep : sopSteps)
            {
                if (sopStep.getStepNo() != null)
                {
                    sopStepByNo.put(sopStep.getStepNo(), sopStep);
                }
                if (StringUtils.isNotBlank(sopStep.getExpectedEvent()))
                {
                    sopStepByEvent.put(sopStep.getExpectedEvent(), sopStep);
                }
            }
            for (SopRuntimeStepUpdate runtimeStep : runtimeSteps)
            {
                if (runtimeStep.getStepNo() == null)
                {
                    continue;
                }
                SopStep matchedSopStep = sopStepByNo.get(runtimeStep.getStepNo());
                if (matchedSopStep == null && StringUtils.isNotBlank(runtimeStep.getExpectedEvent()))
                {
                    matchedSopStep = sopStepByEvent.get(runtimeStep.getExpectedEvent());
                }
                if (matchedSopStep == null || matchedSopStep.getStepId() == null)
                {
                    continue;
                }
                SopTaskStep taskStep = new SopTaskStep();
                taskStep.setTaskId(task.getTaskId());
                taskStep.setStepId(matchedSopStep.getStepId());
                taskStep.setStepNo(runtimeStep.getStepNo());
                taskStep.setStepName(StringUtils.defaultIfBlank(runtimeStep.getStepName(), matchedSopStep.getStepName()));
                taskStep.setExpectedEvent(StringUtils.defaultIfBlank(runtimeStep.getExpectedEvent(), matchedSopStep.getExpectedEvent()));
                taskStep.setRequiredConfidence(matchedSopStep.getRequiredConfidence());
                taskStep.setStepStatus(StringUtils.defaultIfBlank(runtimeStep.getStepStatus(), "PENDING"));
                taskStep.setCreateBy("runtime");
                taskStepMapper.insertSopTaskStep(taskStep);
                created.add(taskStep);
            }
        }

        if (!created.isEmpty())
        {
            return fetchTaskSteps(task.getTaskId());
        }

        for (SopStep sopStep : sopSteps)
        {
            SopTaskStep taskStep = new SopTaskStep();
            taskStep.setTaskId(task.getTaskId());
            taskStep.setStepId(sopStep.getStepId());
            taskStep.setStepNo(sopStep.getStepNo());
            taskStep.setStepName(sopStep.getStepName());
            taskStep.setExpectedEvent(sopStep.getExpectedEvent());
            taskStep.setRequiredConfidence(sopStep.getRequiredConfidence());
            taskStep.setStepStatus("PENDING");
            taskStep.setCreateBy("runtime");
            taskStepMapper.insertSopTaskStep(taskStep);
        }
        return fetchTaskSteps(task.getTaskId());
    }

    private void mergeTaskSteps(List<SopTaskStep> taskSteps, List<SopRuntimeStepUpdate> updates)
    {
        if (updates == null || updates.isEmpty())
        {
            return;
        }
        for (SopRuntimeStepUpdate update : updates)
        {
            if (update.getStepNo() == null)
            {
                continue;
            }
            SopTaskStep taskStep = taskSteps.stream()
                .filter(item -> Objects.equals(item.getStepNo(), update.getStepNo()))
                .findFirst()
                .orElse(null);
            if (taskStep == null)
            {
                continue;
            }
            if (StringUtils.isNotBlank(update.getStepName()))
            {
                taskStep.setStepName(update.getStepName());
            }
            if (StringUtils.isNotBlank(update.getExpectedEvent()))
            {
                taskStep.setExpectedEvent(update.getExpectedEvent());
            }
            if (StringUtils.isNotBlank(update.getStepStatus()))
            {
                taskStep.setStepStatus(update.getStepStatus());
            }
            if (update.getSnapshotUrl() != null)
            {
                taskStep.setSnapshotUrl(update.getSnapshotUrl());
            }
            if (update.getClipUrl() != null)
            {
                taskStep.setClipUrl(update.getClipUrl());
            }
            if (update.getClipStartMs() != null)
            {
                taskStep.setClipStartMs(update.getClipStartMs());
            }
            if (update.getClipEndMs() != null)
            {
                taskStep.setClipEndMs(update.getClipEndMs());
            }
            if (update.getJudgeResult() != null)
            {
                taskStep.setJudgeResult(update.getJudgeResult());
            }
            if (update.getJudgeMessage() != null)
            {
                taskStep.setJudgeMessage(update.getJudgeMessage());
            }
            if (update.getPassTime() != null)
            {
                taskStep.setPassTime(update.getPassTime());
            }
            taskStep.setUpdateBy("runtime");
            taskStepMapper.updateSopTaskStep(taskStep);
        }
    }

    private List<SopTaskStep> fetchTaskSteps(Long taskId)
    {
        SopTaskStep query = new SopTaskStep();
        query.setTaskId(taskId);
        return taskStepMapper.selectSopTaskStepList(query).stream()
            .sorted(Comparator.comparing(SopTaskStep::getStepNo, Comparator.nullsLast(Integer::compareTo)))
            .collect(Collectors.toList());
    }

    private List<SopStep> fetchSopSteps(Long sopId)
    {
        SopStep query = new SopStep();
        query.setSopId(sopId);
        query.setStatus("0");
        return stepMapper.selectSopStepList(query);
    }

    private SopDetectionTask findCurrentTask(Long productId, Long sopId)
    {
        if (productId == null || sopId == null)
        {
            return null;
        }
        SopDetectionTask query = new SopDetectionTask();
        query.setProductId(productId);
        query.setSopId(sopId);
        List<SopDetectionTask> tasks = taskMapper.selectSopDetectionTaskList(query);
        if (tasks == null || tasks.isEmpty())
        {
            return null;
        }
        List<SopDetectionTask> activeTasks = tasks.stream()
            .filter(item -> isActiveTask(item.getTaskStatus()))
            .collect(Collectors.toList());
        if (!activeTasks.isEmpty())
        {
            return activeTasks.stream()
                .max(Comparator.comparing(item -> item.getTaskId() == null ? 0L : item.getTaskId()))
                .orElse(null);
        }
        return tasks.stream()
            .max(Comparator.comparing(item -> item.getTaskId() == null ? 0L : item.getTaskId()))
            .orElse(null);
    }

    private boolean isActiveTask(String taskStatus)
    {
        return StringUtils.equalsAnyIgnoreCase(taskStatus, "CREATED", "RUNNING", "STOPPED");
    }

    private SopDetectionTask resolveSessionTask(SopRuntimeSessionRequest request)
    {
        SopDetectionTask task = null;
        if (StringUtils.isNotBlank(request.getTaskCode()))
        {
            task = taskMapper.selectSopDetectionTaskByTaskCode(request.getTaskCode());
        }
        if (task == null)
        {
            task = findCurrentTask(request.getProductId(), request.getSopId());
        }
        if (task == null)
        {
            return null;
        }
        if (!Objects.equals(task.getProductId(), request.getProductId()) || !Objects.equals(task.getSopId(), request.getSopId()))
        {
            return null;
        }
        return task;
    }

    private void cancelActiveTasks(Long productId, Long sopId, String runtimeMessage, String preferredTaskCode)
    {
        SopDetectionTask current = preferredTaskCode == null ? findCurrentTask(productId, sopId) : taskMapper.selectSopDetectionTaskByTaskCode(preferredTaskCode);
        if (current != null && Objects.equals(current.getProductId(), productId) && Objects.equals(current.getSopId(), sopId) && isActiveTask(current.getTaskStatus()))
        {
            current.setTaskStatus("CANCELLED");
            current.setEndTime(new Date());
            current.setRuntimeMode("RESET");
            current.setRuntimeMessage(runtimeMessage);
            current.setUpdateBy("runtime");
            taskMapper.updateSopDetectionTask(current);
        }

        SopDetectionTask query = new SopDetectionTask();
        query.setProductId(productId);
        query.setSopId(sopId);
        List<SopDetectionTask> tasks = taskMapper.selectSopDetectionTaskList(query);
        for (SopDetectionTask task : tasks)
        {
            if (task.getTaskId() == null || !isActiveTask(task.getTaskStatus()))
            {
                continue;
            }
            if (current != null && Objects.equals(current.getTaskId(), task.getTaskId()))
            {
                continue;
            }
            task.setTaskStatus("CANCELLED");
            task.setEndTime(new Date());
            task.setRuntimeMode("RESET");
            task.setRuntimeMessage(runtimeMessage);
            task.setUpdateBy("runtime");
            taskMapper.updateSopDetectionTask(task);
        }
    }

    private void armTask(SopDetectionTask task)
    {
        Date now = new Date();
        task.setTaskStatus("RUNNING");
        task.setRuntimeMode("ARMED");
        task.setRuntimeMessage("Detection armed, waiting for box_opened");
        if (task.getStartTime() == null)
        {
            task.setStartTime(now);
        }
        task.setEndTime(null);
        task.setUpdateBy("runtime");
        taskMapper.updateSopDetectionTask(task);
    }

    private void stopTask(SopDetectionTask task)
    {
        task.setTaskStatus("STOPPED");
        task.setRuntimeMode("STOPPED");
        task.setRuntimeMessage("Detection stopped, preview only");
        task.setUpdateBy("runtime");
        taskMapper.updateSopDetectionTask(task);
    }

    private SopDetectionTask createTask(SopRuntimeSessionRequest request)
    {
        SopProduct product = productMapper.selectSopProductByProductId(request.getProductId());
        SopProcess process = processMapper.selectSopProcessBySopId(request.getSopId());
        List<SopStep> sopSteps = fetchSopSteps(request.getSopId());

        SopDetectionTask task = new SopDetectionTask();
        task.setTaskCode(buildTaskCode(product));
        task.setProductId(request.getProductId());
        task.setSopId(request.getSopId());
        task.setStationCode(StringUtils.defaultIfBlank(request.getStationCode(), "STATION-01"));
        task.setCameraCode(StringUtils.defaultIfBlank(request.getCameraCode(), "MV-CS050-10UC"));
        task.setCurrentStepNo(1);
        task.setTaskStatus("CREATED");
        task.setOperatorName(StringUtils.defaultIfBlank(request.getOperatorName(), "runtime"));
        task.setRuntimeMode("READY");
        task.setRuntimeMessage("Session ready, waiting for bridge");
        task.setCreateBy("runtime");
        if (product != null)
        {
            task.setProductCode(product.getProductCode());
            task.setProductName(product.getProductName());
        }
        if (process != null)
        {
            task.setSopName(process.getSopName());
        }
        taskMapper.insertSopDetectionTask(task);

        for (SopStep sopStep : sopSteps)
        {
            SopTaskStep taskStep = new SopTaskStep();
            taskStep.setTaskId(task.getTaskId());
            taskStep.setStepId(sopStep.getStepId());
            taskStep.setStepNo(sopStep.getStepNo());
            taskStep.setStepName(sopStep.getStepName());
            taskStep.setExpectedEvent(sopStep.getExpectedEvent());
            taskStep.setRequiredConfidence(sopStep.getRequiredConfidence());
            taskStep.setStepStatus("PENDING");
            taskStep.setCreateBy("runtime");
            taskStepMapper.insertSopTaskStep(taskStep);
        }

        return taskMapper.selectSopDetectionTaskByTaskId(task.getTaskId());
    }

    private String buildTaskCode(SopProduct product)
    {
        String productCode = product != null && StringUtils.isNotBlank(product.getProductCode())
            ? product.getProductCode()
            : "SOP";
        String safeCode = productCode.replaceAll("[^A-Za-z0-9]+", "_");
        return "TASK_" + safeCode + "_" + new SimpleDateFormat("yyyyMMddHHmmssSSS").format(new Date());
    }

    private SopRuntimeView buildRuntimeView(SopDetectionTask task)
    {
        SopRuntimeView view = new SopRuntimeView();
        view.setTask(enrichTask(task));
        view.setSteps(fetchTaskSteps(task.getTaskId()));

        SopDetectionEvent eventQuery = new SopDetectionEvent();
        eventQuery.setTaskCode(task.getTaskCode());
        List<SopDetectionEvent> recentEvents = eventMapper.selectSopDetectionEventList(eventQuery);
        if (recentEvents.size() > 20)
        {
            recentEvents = recentEvents.subList(0, 20);
        }
        view.setRecentEvents(recentEvents);

        SopAlarmRecord alarmQuery = new SopAlarmRecord();
        alarmQuery.setTaskCode(task.getTaskCode());
        List<SopAlarmRecord> recentAlarms = alarmMapper.selectSopAlarmRecordList(alarmQuery);
        if (recentAlarms.size() > 20)
        {
            recentAlarms = recentAlarms.subList(0, 20);
        }
        view.setRecentAlarms(recentAlarms);
        return view;
    }

    private SopDetectionTask enrichTask(SopDetectionTask task)
    {
        if (task == null)
        {
            return null;
        }
        if (StringUtils.isBlank(task.getSopName()) && task.getSopId() != null)
        {
            SopProcess process = processMapper.selectSopProcessBySopId(task.getSopId());
            if (process != null)
            {
                task.setSopName(process.getSopName());
            }
        }
        return task;
    }
}
