package com.ruoyi.system.domain.sop;

public class SopRuntimeSessionRequest
{
    private Long productId;
    private Long sopId;
    private String stationCode;
    private String cameraCode;
    private String operatorName;
    private String taskCode;

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

    public String getOperatorName()
    {
        return operatorName;
    }

    public void setOperatorName(String operatorName)
    {
        this.operatorName = operatorName;
    }

    public String getTaskCode()
    {
        return taskCode;
    }

    public void setTaskCode(String taskCode)
    {
        this.taskCode = taskCode;
    }
}
