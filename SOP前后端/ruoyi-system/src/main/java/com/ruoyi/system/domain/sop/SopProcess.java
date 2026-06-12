package com.ruoyi.system.domain.sop;

import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;
import com.ruoyi.common.annotation.Excel;
import com.ruoyi.common.core.domain.BaseEntity;

/**
 * SOP process object sop_process
 *
 * @author ruoyi
 */
public class SopProcess extends BaseEntity
{
    private static final long serialVersionUID = 1L;

    /** SOP ID */
    private Long sopId;

    /** SOP code */
    @Excel(name = "SOP编码")
    private String sopCode;

    /** SOP name */
    @Excel(name = "SOP名称")
    private String sopName;

    /** Product ID */
    @Excel(name = "产品ID")
    private Long productId;

    /** Product code */
    @Excel(name = "产品编码")
    private String productCode;

    /** Product name */
    @Excel(name = "产品名称")
    private String productName;

    /** SOP version */
    @Excel(name = "版本")
    private String version;

    /** Status: 0 normal, 1 disabled */
    @Excel(name = "状态", readConverterExp = "0=正常,1=停用")
    private String status;

    public Long getSopId()
    {
        return sopId;
    }

    public void setSopId(Long sopId)
    {
        this.sopId = sopId;
    }

    public String getSopCode()
    {
        return sopCode;
    }

    public void setSopCode(String sopCode)
    {
        this.sopCode = sopCode;
    }

    public String getSopName()
    {
        return sopName;
    }

    public void setSopName(String sopName)
    {
        this.sopName = sopName;
    }

    public Long getProductId()
    {
        return productId;
    }

    public void setProductId(Long productId)
    {
        this.productId = productId;
    }

    public String getProductCode()
    {
        return productCode;
    }

    public void setProductCode(String productCode)
    {
        this.productCode = productCode;
    }

    public String getProductName()
    {
        return productName;
    }

    public void setProductName(String productName)
    {
        this.productName = productName;
    }

    public String getVersion()
    {
        return version;
    }

    public void setVersion(String version)
    {
        this.version = version;
    }

    public String getStatus()
    {
        return status;
    }

    public void setStatus(String status)
    {
        this.status = status;
    }

    @Override
    public String toString()
    {
        return new ToStringBuilder(this, ToStringStyle.MULTI_LINE_STYLE)
            .append("sopId", getSopId())
            .append("sopCode", getSopCode())
            .append("sopName", getSopName())
            .append("productId", getProductId())
            .append("productCode", getProductCode())
            .append("productName", getProductName())
            .append("version", getVersion())
            .append("status", getStatus())
            .append("createBy", getCreateBy())
            .append("createTime", getCreateTime())
            .append("updateBy", getUpdateBy())
            .append("updateTime", getUpdateTime())
            .append("remark", getRemark())
            .toString();
    }
}
