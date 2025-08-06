const hre = require("hardhat");
const fs = require('fs');
const path = require('path');

async function main() {
    const [deployer] = await hre.ethers.getSigners();
    console.log("Deploying contracts with the account:", deployer.address);

    // Deploy Loan Token
    const LoanToken = await hre.ethers.getContractFactory("LoanToken");
    const loanToken = await LoanToken.deploy(deployer.address);
    await loanToken.waitForDeployment();
    const loanTokenAddress = await loanToken.getAddress();
    console.log("✅ LoanToken deployed to:", loanTokenAddress);

    // Deploy Lending Pool
    const LendingPool = await hre.ethers.getContractFactory("LendingPool");
    const lendingPool = await LendingPool.deploy(loanTokenAddress, deployer.address);
    await lendingPool.waitForDeployment();
    const lendingPoolAddress = await lendingPool.getAddress();
    console.log("✅ LendingPool deployed to:", lendingPoolAddress);

    // Transfer initial token supply to the Lending Pool for it to distribute
    console.log("Transferring initial FLT supply to LendingPool...");
    const initialSupply = await loanToken.INITIAL_SUPPLY();
    const tx = await loanToken.transfer(lendingPoolAddress, initialSupply);
    await tx.wait();
    console.log("✅ Supply transferred successfully.");

    // --- Copy ABI files to backend and frontend ---
    console.log("Copying ABI files...");
    copyAbiFiles();
}

function copyAbiFiles() {
    const contractsDir = path.join(__dirname, "../../backend/contracts");

    if (!fs.existsSync(contractsDir)) {
        fs.mkdirSync(contractsDir, { recursive: true });
    }

    const loanTokenArtifact = hre.artifacts.readArtifactSync("LoanToken");
    fs.writeFileSync(
        path.join(contractsDir, "LoanToken.json"),
        JSON.stringify(loanTokenArtifact, null, 2)
    );

    const lendingPoolArtifact = hre.artifacts.readArtifactSync("LendingPool");
    fs.writeFileSync(
        path.join(contractsDir, "LendingPool.json"),
        JSON.stringify(lendingPoolArtifact, null, 2)
    );

    console.log("✅ ABI files copied to backend/contracts");
}


main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});