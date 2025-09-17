'use client'
import React from 'react'
import { LogoIcon, CloseIcon, NewIcon, SearchIcon, OptionIcon } from '../Icon'

const SideBar = () => {
    //Handler function
    const handleClickLogo = () => {
        console.log("Click logo")
    }
    const handleClickClose = () => {
        console.log("Click close button")
    }
    const handleClickNewChat = () => {
        console.log("Click new chat")
    }
    const handleClickSearch = () => {
        console.log("Click search")
    }
    const handleClickOption = () => {
        console.log("Click option")
    }
    const handleClickProfile = () => {
        console.log("Click profile")
    }
    return (
            <div className='flex flex-col w-full h-screen text-white px-2 overflow-y-auto bg-sider'>
                <div className='flex flex-row justify-between py-5'>
                    <div className='p-2 cursor-pointer hover:bg-hover-button  hover:rounded-lg flex items-center justify-center'>
                        <button 
                            className='cursor-pointer'
                            onClick={handleClickLogo}
                        >
                            <div className='w-6 h-6'>
                                <LogoIcon/>
                            </div>
                        </button>
                    </div>
                    <div className='p-2 cursor-pointer hover:bg-hover-button  hover:rounded-lg  flex items-center justify-center'>
                        <button 
                            className='cursor-pointer'
                            onClick={handleClickClose}
                        >
                            <div className='w-6 h-6'>
                                <CloseIcon/>
                            </div>
                        </button>
                    </div>
                </div>
                <div className='flex flex-col gap-2 py-2'>
                    {/* Bam vao la chuyen sang dia chi url moi reset het */}
                    <div 
                        className='flex flex-row gap-3 p-2 items-center cursor-pointer hover:bg-hover-button hover:rounded-lg '
                        onClick={handleClickNewChat}
                    > 
                        <div className='w-6 h-6 cursor-pointer'>
                            <NewIcon/>
                        </div>
                        <span className=''>New chat</span>
                    </div>
                    {/* Bam nao la hien popup co thanh search bar */}
                    <div 
                        className='flex flex-row gap-3 p-2 items-center cursor-pointer hover:bg-hover-button  hover:rounded-lg -hover-button'
                        onClick={handleClickSearch}
                    >
                        <div className='w-6 h-6 cursor-pointer'>
                            <SearchIcon/>
                        </div>
                        <span>Search chats</span>
                    </div>
                </div>
                {/* Chats */}
                <div className='flex flex-col py-5'>
                    {/* Ten title cua doan chat */}
                    <span className='px-1 text-gray-400 pb-4'>Chats</span>
                    <div className='flex flex-col gap-3'>
                        <div className='flex flex-row p-2 justify-between hover:bg-hover-button hover:rounded-lg cursor-pointer'>
                            <span className='line-clamp-1'>Bật tự động xuống dòng</span>
                            <div 
                                className='w-6 h-6 ml-4'
                                onClick={handleClickOption}
                            >
                                <OptionIcon/>
                            </div>
                        </div>
                    </div>
                    <div className='flex flex-col gap-3'>
                        <div className='flex flex-row p-2 justify-between hover:bg-hover-button  hover:rounded-lg  cursor-pointer'>
                            <span className='line-clamp-1'>Bật tự động xuống dòng</span>
                            <div className='w-6 h-6 ml-4'>
                                <OptionIcon/>
                            </div>
                        </div>
                    </div>
                    <div className='flex flex-col gap-3'>
                        <div className='flex flex-row p-2 justify-between hover:bg-hover-button  hover:rounded-lg  cursor-pointer'>
                            <span className='line-clamp-1'>Bật tự động xuống dòng</span>
                            <div className='w-6 h-6 ml-4'>
                                <OptionIcon/>
                            </div>
                        </div>
                    </div>
                    <div className='flex flex-col gap-3'>
                        <div className='flex flex-row p-2 justify-between hover:bg-hover-button  hover:rounded-lg  cursor-pointer'>
                            <span className='line-clamp-1'>Bật tự động xuống dòng</span>
                            <div className='w-6 h-6 ml-4'>
                                <OptionIcon/>
                            </div>
                        </div>
                    </div>
                    <div className='flex flex-col gap-3'>
                        <div className='flex flex-row p-2 justify-between hover:bg-hover-button  hover:rounded-lg  cursor-pointer'>
                            <span className='line-clamp-1'>Bật tự động xuống dòng</span>
                            <div className='w-6 h-6 ml-4'>
                                <OptionIcon/>
                            </div>
                        </div>
                    </div>
                    <div className='flex flex-col gap-3'>
                        <div className='flex flex-row p-2 justify-between hover:bg-hover-button  hover:rounded-lg  cursor-pointer'>
                            <span className='line-clamp-1'>Bật tự động xuống dòng</span>
                            <div className='w-6 h-6 ml-4'>
                                <OptionIcon/>
                            </div>
                        </div>
                    </div>
                    <div className='flex flex-col gap-3'>
                        <div className='flex flex-row p-2 justify-between hover:bg-hover-button  hover:rounded-lg  cursor-pointer'>
                            <span className='line-clamp-1'>Bật tự động xuống dòng</span>
                            <div className='w-6 h-6 ml-4'>
                                <OptionIcon/>
                            </div>
                        </div>
                    </div>
                    <div className='flex flex-col gap-3'>
                        <div className='flex flex-row p-2 justify-between hover:bg-hover-button  hover:rounded-lg  cursor-pointer'>
                            <span className='line-clamp-1'>Bật tự động xuống dòng</span>
                            <div className='w-6 h-6 ml-4'>
                                <OptionIcon/>
                            </div>
                        </div>
                    </div>
                    <div className='flex flex-col gap-3'>
                        <div className='flex flex-row p-2 justify-between hover:bg-hover-button  hover:rounded-lg  cursor-pointer'>
                            <span className='line-clamp-1'>Bật tự động xuống dòng</span>
                            <div className='w-6 h-6 ml-4'>
                                <OptionIcon/>
                            </div>
                        </div>
                    </div>
                    <div className='flex flex-col gap-3'>
                        <div className='flex flex-row p-2 justify-between hover:bg-hover-button  hover:rounded-lg  cursor-pointer'>
                            <span className='line-clamp-1'>Bật tự động xuống dòng</span>
                            <div className='w-6 h-6 ml-4'>
                                <OptionIcon/>
                            </div>
                        </div>
                    </div>
                    <div className='flex flex-col gap-3'>
                        <div className='flex flex-row p-2 justify-between hover:bg-hover-button hover:rounded-lg  cursor-pointer'>
                            <span className='line-clamp-1'>Bật tự động xuống dòng</span>
                            <div className='w-6 h-6 ml-4'>
                                <OptionIcon/>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="sticky bottom-0 bg-sider text-white h-[8%]">
                    <div 
                        className='flex flex-row gap-4 h-full items-center py-2 hover:bg-hover-button  hover:rounded-lg  cursor-pointer p-2'
                        onClick={handleClickProfile}
                    >
                        <div className='w-6 h-6'>
                            <LogoIcon/>
                        </div>
                        <span>Nguyễn Đức Trí</span>
                    </div>
                </div>
            </div>
    )
}

export default SideBar
