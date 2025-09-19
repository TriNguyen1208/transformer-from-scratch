import Image from "next/image";
import SideBar from "./components/SideBar";
import MainContent from "./components/MainContent";
export default function Home() {
    return (
        <div>
            <div className="flex flex-row">
                <div className="w-[20%]">
                    <SideBar/>  
                </div>
                <div className="w-[80%]">
                    <MainContent/>
                </div>
            </div>

        </div>
    );
}
