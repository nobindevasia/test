using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.Core.Interfaces
{
    public interface IConfigManager
    {
        ModelConfig LoadConfiguration(string configPath);
        void ValidateConfiguration(ModelConfig config);
    }
}

