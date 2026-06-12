package com.ruoyi.system.domain.sop;

import java.util.List;

public class SopRuntimeSyncRequest
{
    private String taskCode;
    private Long productId;
    private Long sopId;
    private String stationCode;
    private String cameraCode;
    private Integer currentStepNo;
    private String taskStatus;
    private String operatorName;
    private String previewStreamUrl;
    private String latestFrameUrl;
    private String runtimeMode;
    private String runtimeMessage;
    private Double runtimeFps;
    private List<SopRuntimeStepUpdate> steps;
    private SopDetectionEvent event;

    public String getTaskCode()
    {
        return taskCode;
    }

    public void setTaskCode(String taskCode)
    {
        this.taskCode = taskCode;
    }

    public Long getProductId()
    {
        return productId;
    }

    public void setProductId(Long productId)
    {
        this.productId = productId;
    }

    public Long getSopId()
    {
        return sopId;
    }

    public void setSopId(Long sopId)
    {
        this.sopId = sopId;
    }

    public String getStationCode()
    {
        return stationCode;
    }

    public void setStationCode(String stationCode)
    {
        this.stationCode = stationCode;
    }

    public String getCameraCode()
    {
        return cameraCode;
    }

    public void setCameraCode(String cameraCode)
    {
        this.cameraCode = cameraCode;
    }

    public Integer getCurrentStepNo()
    {
        return currentStepNo;
    }

    public void setCurrentStepNo(Integer currentStepNo)
    {
        this.currentStepNo = currentStepNo;
    }

    public String getTaskStatus()
    {
        return taskStatus;
    }

    public void setTaskStatus(String taskStatus)
    {
        this.taskStatus = taskStatus;
    }

    public String getOperatorName()
    {
        return operatorName;
    }

    public void setOperatorName(String operatorName)
    {
        this.operatorName = operatorName;
    }

    public String getPreviewStreamUrl()
    {
        return previewStreamUrl;
    }

    public void setPreviewStreamUrl(String previewStreamUrl)
    {
        this.previewStreamUrl = previewStreamUrl;
    }

    public String getLatestFrameUrl()
    {
        return latestFrameUrl;
    }

    public void setLatestFrameUrl(String latestFrameUrl)
    {
        this.latestFrameUrl = latestFrameUrl;
    }

    public String getRuntimeMode()
    {
        return runtimeMode;
    }

    public void setRuntimeMode(String runtimeMode)
    {
        this.runtimeMode = runtimeMode;
    }

    public String getRuntimeMessage()
    {
        return runtimeMessage;
    }

    public void setRuntimeMessage(String runtimeMessage)
    {
        this.runtimeMessage = runtimeMessage;
    }

    public Double getRuntimeFps()
    {
        return runtimeFps;
    }

    public void setRuntimeFps(Double runtimeFps)
    {
        this.runtimeFps = runtimeFps;
    }

    public List<SopRuntimeStepUpdate> getSteps()
    {
        return steps;
    }

    public void setSteps(List<SopRuntimeStepUpdate> steps)
    {
        this.steps = steps;
    }

    public SopDetectionEvent getEvent()
    {
        return event;
    }

    public void setEvent(SopDetectionEvent event)
    {
        this.event = event;
    }
}
