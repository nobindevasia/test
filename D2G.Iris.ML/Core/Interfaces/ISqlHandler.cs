
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Enums;

namespace D2G.Iris.ML.Core.Interfaces
{
    public interface ISqlHandler
    {
        void Connect(DatabaseConfig dbConfig);
        string GetConnectionString();
        void SaveProcessedDataToTable(
            string tableName,
            List<Dictionary<string, object>> processedData,
            string[] featureNames,
            string targetField,
            ModelType modelType);
    }
}